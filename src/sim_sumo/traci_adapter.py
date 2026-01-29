"""SUMO/TraCI adapter for Stage 2 validation.

This module implements the TraCI interface for running trained policies
in SUMO simulation. It handles:
- SUMO connection management
- Vehicle routing and control
- Real-time travel time measurement
- Passenger state management synchronized with SUMO dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import os
import sys
import time

import numpy as np

LOG = logging.getLogger(__name__)

TRACI_AVAILABLE = False
try:
    if "SUMO_HOME" in os.environ:
        tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)
    import traci
    from traci import constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    traci = None
    tc = None
    LOG.warning("TraCI not available. Install SUMO and set SUMO_HOME environment variable.")


@dataclass
class TraCIConfig:
    """Configuration for TraCI connection."""
    
    sumo_cfg_path: str = ""
    sumo_binary: str = "sumo"
    sumo_gui: bool = False
    sumo_port: int = 8813
    sumo_seed: int = 7
    sumo_step_length: float = 1.0
    sumo_start_time: float = 0.0
    sumo_end_time: float = 36000.0
    sumo_warmup_steps: int = 100
    sumo_additional_args: List[str] = None
    
    label: str = "default"
    
    def __post_init__(self):
        if self.sumo_additional_args is None:
            self.sumo_additional_args = []


class TraCIConnection:
    """Manages TraCI connection to SUMO simulation."""
    
    def __init__(self, config: TraCIConfig) -> None:
        self.config = config
        self.connected = False
        self.current_time = 0.0
        self.step_count = 0
        
        self._vehicles: Dict[str, Dict[str, Any]] = {}
        self._pending_routes: Dict[str, List[str]] = {}
        
    def connect(self) -> None:
        """Start SUMO and establish TraCI connection."""
        if not TRACI_AVAILABLE:
            raise RuntimeError("TraCI not available. Install SUMO and set SUMO_HOME.")
        
        if self.connected:
            LOG.warning("Already connected to SUMO")
            return
        
        binary = self.config.sumo_binary
        if self.config.sumo_gui:
            binary = binary.replace("sumo", "sumo-gui")
        
        sumo_cmd = [
            binary,
            "-c", self.config.sumo_cfg_path,
            "--step-length", str(self.config.sumo_step_length),
            "--seed", str(self.config.sumo_seed),
            "--start",
            "--quit-on-end",
        ]
        
        if self.config.sumo_start_time > 0:
            sumo_cmd.extend(["--begin", str(self.config.sumo_start_time)])
        if self.config.sumo_end_time > 0:
            sumo_cmd.extend(["--end", str(self.config.sumo_end_time)])
        
        sumo_cmd.extend(self.config.sumo_additional_args)
        
        LOG.info("Starting SUMO: %s", " ".join(sumo_cmd))
        
        traci.start(sumo_cmd, label=self.config.label, port=self.config.sumo_port)
        self.connected = True
        self.current_time = traci.simulation.getTime()
        self.step_count = 0
        
        LOG.info("Connected to SUMO at time %.2f", self.current_time)
    
    def disconnect(self) -> None:
        """Close TraCI connection."""
        if not self.connected:
            return
        
        try:
            traci.close()
        except Exception as e:
            LOG.warning("Error closing TraCI: %s", e)
        
        self.connected = False
        LOG.info("Disconnected from SUMO after %d steps", self.step_count)
    
    def step(self, duration: float = None) -> float:
        """Advance simulation by one step or specified duration."""
        if not self.connected:
            raise RuntimeError("Not connected to SUMO")
        
        if duration is None:
            traci.simulationStep()
            self.step_count += 1
        else:
            target_time = self.current_time + duration
            while self.current_time < target_time:
                traci.simulationStep()
                self.step_count += 1
                self.current_time = traci.simulation.getTime()
        
        self.current_time = traci.simulation.getTime()
        return self.current_time
    
    def warmup(self, steps: int = None) -> None:
        """Run warmup steps to populate network with background traffic."""
        if steps is None:
            steps = self.config.sumo_warmup_steps
        
        LOG.info("Running %d warmup steps...", steps)
        for _ in range(steps):
            self.step()
        LOG.info("Warmup complete at time %.2f", self.current_time)
    
    def get_time(self) -> float:
        """Get current simulation time."""
        if self.connected:
            self.current_time = traci.simulation.getTime()
        return self.current_time
    
    def add_vehicle(
        self,
        vehicle_id: str,
        route_id: str = None,
        vehicle_type: str = "DEFAULT_VEHTYPE",
        depart_time: float = None,
        depart_pos: str = "base",
        depart_speed: str = "0",
    ) -> bool:
        """Add a vehicle to the simulation."""
        if not self.connected:
            raise RuntimeError("Not connected to SUMO")
        
        if depart_time is None:
            depart_time = self.current_time
        
        try:
            if route_id and route_id in [r for r in traci.route.getIDList()]:
                traci.vehicle.add(
                    vehID=vehicle_id,
                    routeID=route_id,
                    typeID=vehicle_type,
                    depart=str(depart_time),
                    departPos=depart_pos,
                    departSpeed=depart_speed,
                )
            else:
                LOG.warning("Route %s not found, vehicle %s not added", route_id, vehicle_id)
                return False
            
            self._vehicles[vehicle_id] = {
                "added_time": self.current_time,
                "route_id": route_id,
                "type": vehicle_type,
            }
            return True
        except traci.TraCIException as e:
            LOG.error("Failed to add vehicle %s: %s", vehicle_id, e)
            return False
    
    def set_vehicle_route(self, vehicle_id: str, edge_ids: List[str]) -> bool:
        """Set route for a vehicle (list of edge IDs)."""
        if not self.connected:
            raise RuntimeError("Not connected to SUMO")
        
        if vehicle_id not in traci.vehicle.getIDList():
            LOG.warning("Vehicle %s not in simulation", vehicle_id)
            return False
        
        try:
            traci.vehicle.setRoute(vehicle_id, edge_ids)
            return True
        except traci.TraCIException as e:
            LOG.error("Failed to set route for %s: %s", vehicle_id, e)
            return False
    
    def set_vehicle_stop(
        self,
        vehicle_id: str,
        edge_id: str,
        pos: float = 1.0,
        duration: float = 60.0,
        flags: int = 0,
    ) -> bool:
        """Set a stop for a vehicle at specified edge."""
        if not self.connected:
            raise RuntimeError("Not connected to SUMO")
        
        try:
            traci.vehicle.setStop(
                vehID=vehicle_id,
                edgeID=edge_id,
                pos=pos,
                laneIndex=0,
                duration=duration,
                flags=flags,
            )
            return True
        except traci.TraCIException as e:
            LOG.error("Failed to set stop for %s at %s: %s", vehicle_id, edge_id, e)
            return False
    
    def resume_vehicle(self, vehicle_id: str) -> bool:
        """Resume a stopped vehicle."""
        if not self.connected:
            raise RuntimeError("Not connected to SUMO")
        
        try:
            traci.vehicle.resume(vehicle_id)
            return True
        except traci.TraCIException as e:
            LOG.error("Failed to resume vehicle %s: %s", vehicle_id, e)
            return False
    
    def get_vehicle_position(self, vehicle_id: str) -> Optional[Tuple[float, float]]:
        """Get vehicle position (x, y) in simulation coordinates."""
        if not self.connected:
            return None
        
        if vehicle_id not in traci.vehicle.getIDList():
            return None
        
        try:
            return traci.vehicle.getPosition(vehicle_id)
        except traci.TraCIException:
            return None
    
    def get_vehicle_edge(self, vehicle_id: str) -> Optional[str]:
        """Get current edge ID for a vehicle."""
        if not self.connected:
            return None
        
        if vehicle_id not in traci.vehicle.getIDList():
            return None
        
        try:
            return traci.vehicle.getRoadID(vehicle_id)
        except traci.TraCIException:
            return None
    
    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Get current speed of a vehicle (m/s)."""
        if not self.connected:
            return 0.0
        
        if vehicle_id not in traci.vehicle.getIDList():
            return 0.0
        
        try:
            return traci.vehicle.getSpeed(vehicle_id)
        except traci.TraCIException:
            return 0.0
    
    def is_vehicle_stopped(self, vehicle_id: str) -> bool:
        """Check if vehicle is currently stopped."""
        if not self.connected:
            return False
        
        if vehicle_id not in traci.vehicle.getIDList():
            return False
        
        try:
            return traci.vehicle.isStopped(vehicle_id)
        except traci.TraCIException:
            return False
    
    def get_vehicle_route(self, vehicle_id: str) -> List[str]:
        """Get current route (list of edge IDs) for a vehicle."""
        if not self.connected:
            return []
        
        if vehicle_id not in traci.vehicle.getIDList():
            return []
        
        try:
            return list(traci.vehicle.getRoute(vehicle_id))
        except traci.TraCIException:
            return []
    
    def get_edge_travel_time(self, edge_id: str) -> float:
        """Get current travel time estimate for an edge."""
        if not self.connected:
            return float("inf")
        
        try:
            return traci.edge.getTraveltime(edge_id)
        except traci.TraCIException:
            return float("inf")
    
    def get_route_travel_time(self, edge_ids: List[str]) -> float:
        """Get total travel time for a route (sum of edge travel times)."""
        if not edge_ids:
            return 0.0
        return sum(self.get_edge_travel_time(e) for e in edge_ids)
    
    def find_route(self, from_edge: str, to_edge: str) -> Tuple[List[str], float]:
        """Find shortest route between two edges."""
        if not self.connected:
            return [], float("inf")
        
        try:
            stage = traci.simulation.findRoute(from_edge, to_edge)
            return list(stage.edges), stage.travelTime
        except traci.TraCIException as e:
            LOG.warning("Route not found from %s to %s: %s", from_edge, to_edge, e)
            return [], float("inf")
    
    def get_simulation_end(self) -> bool:
        """Check if simulation has ended."""
        if not self.connected:
            return True
        
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except traci.TraCIException:
            return True


class StopRouteManager:
    """Manages routing between logical stops and SUMO network edges."""
    
    def __init__(
        self,
        stop_to_edge: Dict[int, str],
        stop_coords: Dict[int, Tuple[float, float]],
        traci_conn: TraCIConnection,
    ) -> None:
        self.stop_to_edge = stop_to_edge
        self.stop_coords = stop_coords
        self.traci = traci_conn
        
        self._route_cache: Dict[Tuple[int, int], Tuple[List[str], float]] = {}
    
    def get_stop_edge(self, stop_id: int) -> Optional[str]:
        """Get SUMO edge ID for a logical stop."""
        return self.stop_to_edge.get(int(stop_id))
    
    def find_stop_route(self, from_stop: int, to_stop: int) -> Tuple[List[str], float]:
        """Find route between two logical stops."""
        cache_key = (int(from_stop), int(to_stop))
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        
        from_edge = self.get_stop_edge(from_stop)
        to_edge = self.get_stop_edge(to_stop)
        
        if from_edge is None or to_edge is None:
            LOG.warning("Missing edge mapping for stops %d->%d", from_stop, to_stop)
            return [], float("inf")
        
        route, travel_time = self.traci.find_route(from_edge, to_edge)
        self._route_cache[cache_key] = (route, travel_time)
        return route, travel_time
    
    def clear_cache(self) -> None:
        """Clear route cache."""
        self._route_cache.clear()


class SUMOVehicleController:
    """Controls a single vehicle in SUMO simulation."""
    
    def __init__(
        self,
        vehicle_id: int,
        sumo_vehicle_id: str,
        traci_conn: TraCIConnection,
        route_manager: StopRouteManager,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.sumo_vehicle_id = sumo_vehicle_id
        self.traci = traci_conn
        self.route_manager = route_manager
        
        self.current_stop: Optional[int] = None
        self.target_stop: Optional[int] = None
        self.departure_time: Optional[float] = None
        self.arrival_time: Optional[float] = None
        self.is_moving = False
        
        self.prior_travel_time = 0.0
        self.actual_travel_time = 0.0
        
    def set_initial_stop(self, stop_id: int) -> None:
        """Set vehicle's initial stop location."""
        self.current_stop = int(stop_id)
        self.target_stop = None
        self.is_moving = False
    
    def dispatch_to_stop(self, target_stop: int, prior_travel_sec: float) -> bool:
        """Dispatch vehicle to a target stop."""
        if self.current_stop is None:
            LOG.error("Vehicle %s has no current stop", self.sumo_vehicle_id)
            return False
        
        route, sumo_travel = self.route_manager.find_stop_route(self.current_stop, target_stop)
        if not route:
            LOG.warning("No route from stop %d to %d", self.current_stop, target_stop)
            return False
        
        success = self.traci.set_vehicle_route(self.sumo_vehicle_id, route)
        if not success:
            return False
        
        self.target_stop = int(target_stop)
        self.departure_time = self.traci.get_time()
        self.prior_travel_time = prior_travel_sec
        self.is_moving = True
        
        self.traci.resume_vehicle(self.sumo_vehicle_id)
        
        return True
    
    def check_arrival(self) -> bool:
        """Check if vehicle has arrived at target stop."""
        if not self.is_moving or self.target_stop is None:
            return False
        
        target_edge = self.route_manager.get_stop_edge(self.target_stop)
        if target_edge is None:
            return False
        
        current_edge = self.traci.get_vehicle_edge(self.sumo_vehicle_id)
        is_stopped = self.traci.is_vehicle_stopped(self.sumo_vehicle_id)
        
        if current_edge == target_edge and is_stopped:
            self.arrival_time = self.traci.get_time()
            self.actual_travel_time = self.arrival_time - self.departure_time
            
            self.current_stop = self.target_stop
            self.target_stop = None
            self.is_moving = False
            
            return True
        
        return False
    
    def get_travel_delta(self) -> Dict[str, float]:
        """Get sim-to-real travel time delta for last trip."""
        return {
            "prior_sec": self.prior_travel_time,
            "actual_sec": self.actual_travel_time,
            "delta_sec": self.actual_travel_time - self.prior_travel_time,
        }


class TraCISimulationRunner:
    """Runs a complete SUMO simulation episode with policy control."""
    
    def __init__(
        self,
        traci_config: TraCIConfig,
        sumo_env: Any,
        policy_fn: Callable,
        stop_to_edge: Dict[int, str],
    ) -> None:
        self.traci_config = traci_config
        self.env = sumo_env
        self.policy_fn = policy_fn
        self.stop_to_edge = stop_to_edge
        
        self.traci_conn: Optional[TraCIConnection] = None
        self.route_manager: Optional[StopRouteManager] = None
        self.vehicle_controllers: Dict[int, SUMOVehicleController] = {}
        
        self.episode_metrics: Dict[str, Any] = {}
        
    def setup(self) -> None:
        """Set up TraCI connection and controllers."""
        self.traci_conn = TraCIConnection(self.traci_config)
        self.traci_conn.connect()
        
        self.route_manager = StopRouteManager(
            stop_to_edge=self.stop_to_edge,
            stop_coords=self.env.stop_coords,
            traci_conn=self.traci_conn,
        )
        
        self.traci_conn.warmup()
        
        self.env.reset()
        self.env.set_traci(self.traci_conn)
        
        for vehicle in self.env.vehicles:
            sumo_veh_id = f"minibus_{vehicle.vehicle_id}"
            controller = SUMOVehicleController(
                vehicle_id=vehicle.vehicle_id,
                sumo_vehicle_id=sumo_veh_id,
                traci_conn=self.traci_conn,
                route_manager=self.route_manager,
            )
            controller.set_initial_stop(vehicle.current_stop)
            self.vehicle_controllers[vehicle.vehicle_id] = controller
    
    def teardown(self) -> None:
        """Clean up TraCI connection."""
        if self.traci_conn:
            self.traci_conn.disconnect()
            self.traci_conn = None
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run a complete episode and return metrics."""
        self.setup()
        
        try:
            return self._run_episode_loop(max_steps)
        finally:
            self.teardown()
    
    def _run_episode_loop(self, max_steps: int) -> Dict[str, Any]:
        """Main simulation loop."""
        total_tacc = 0.0
        steps = 0
        
        while steps < max_steps and not self.env.done:
            for vid, controller in self.vehicle_controllers.items():
                if controller.is_moving:
                    if controller.check_arrival():
                        vehicle = self.env.vehicles[vid]
                        vehicle.current_stop = controller.current_stop
                        vehicle.is_moving = False
                        
                        delta = controller.get_travel_delta()
                        self.env.record_travel_delta(
                            src=vehicle.current_stop,
                            dst=controller.target_stop if controller.target_stop else vehicle.current_stop,
                            prior_sec=delta["prior_sec"],
                            actual_sec=delta["actual_sec"],
                        )
                        
                        self._process_arrival(vehicle)
                        
                        if vid not in self.env.ready_vehicle_ids:
                            self.env.ready_vehicle_ids.append(vid)
            
            if self.env.ready_vehicle_ids:
                self.env.active_vehicle_id = self.env.ready_vehicle_ids.pop(0)
            
            if self.env.active_vehicle_id is not None:
                features = self.env.get_feature_batch()
                action = self.policy_fn(features)
                
                if action is not None:
                    vehicle = self.env._get_active_vehicle()
                    controller = self.vehicle_controllers[vehicle.vehicle_id]
                    
                    prior_travel = self.env.get_prior_travel_time(vehicle.current_stop, action)
                    
                    if controller.dispatch_to_stop(action, prior_travel):
                        vehicle.target_stop = action
                        vehicle.is_moving = True
                        self.env.active_vehicle_id = None
                        steps += 1
            
            self._process_pending_requests()
            self._apply_churn()
            
            self.traci_conn.step()
            self.env.current_time = self.traci_conn.get_time()
            
            if self.traci_conn.get_simulation_end():
                break
        
        return self._compile_metrics(total_tacc, steps)
    
    def _process_arrival(self, vehicle) -> None:
        """Process vehicle arrival at a stop (boarding/alighting)."""
        stop_id = vehicle.current_stop
        
        dropped = [p for p in vehicle.onboard if p["dropoff_stop_id"] == stop_id]
        vehicle.onboard = [p for p in vehicle.onboard if p["dropoff_stop_id"] != stop_id]
        
        for pax in dropped:
            pax["status"] = "served"
            self.env.served += 1
            self.env.dropoff_count_by_stop[stop_id] += 1
        
        capacity_left = self.env.config.vehicle_capacity - len(vehicle.onboard)
        queue = self.env.waiting.get(stop_id, [])
        boarded = queue[:capacity_left]
        self.env.waiting[stop_id] = queue[capacity_left:]
        
        for req in boarded:
            req["pickup_time_sec"] = self.env.current_time
            req["t_max_sec"] = self.env.config.mask_alpha * req.get("direct_time_sec", float("inf"))
            req["status"] = "onboard"
            vehicle.onboard.append(req)
            
            wait_sec = self.env.current_time - req["request_time_sec"]
            self.env.acc_wait_time_by_stop[stop_id] += wait_sec
        
        if boarded:
            self.env.service_count_by_stop[stop_id] += len(boarded)
        
        vehicle.visit_counts[stop_id] = vehicle.visit_counts.get(stop_id, 0) + 1
    
    def _process_pending_requests(self) -> None:
        """Add pending requests that have arrived to waiting queues."""
        for req in self.env.requests:
            if req["status"] is not None:
                continue
            if req.get("structural_unserviceable", False):
                continue
            if req["request_time_sec"] <= self.env.current_time:
                req["status"] = "waiting"
                pickup_stop = req["pickup_stop_id"]
                self.env.waiting[pickup_stop].append(req)
    
    def _apply_churn(self) -> None:
        """Apply churn to waiting and onboard passengers."""
        for stop_id, queue in self.env.waiting.items():
            remain = []
            for req in queue:
                wait_sec = self.env.current_time - req["request_time_sec"]
                
                if wait_sec > self.env.config.request_timeout_sec:
                    req["status"] = "churned_waiting"
                    req["cancel_reason"] = "timeout"
                    self.env.waiting_timeouts += 1
                    self.env.canceled_requests.append(req)
                    continue
                
                prob = self.env._waiting_churn_prob(wait_sec)
                if self.env.rng.random() < prob:
                    req["status"] = "churned_waiting"
                    req["cancel_reason"] = "probabilistic_churn"
                    self.env.waiting_churned += 1
                    self.env.canceled_requests.append(req)
                else:
                    remain.append(req)
            
            self.env.waiting[stop_id] = remain
        
        for vehicle in self.env.vehicles:
            remain = []
            for pax in vehicle.onboard:
                pickup_time = pax.get("pickup_time_sec", self.env.current_time)
                elapsed = self.env.current_time - pickup_time
                delay = max(0.0, elapsed - pax.get("direct_time_sec", elapsed))
                
                prob = self.env._onboard_churn_prob(delay)
                if self.env.rng.random() < prob:
                    pax["status"] = "churned_onboard"
                    pax["cancel_reason"] = "onboard_churn"
                    self.env.onboard_churned += 1
                    self.env.canceled_requests.append(pax)
                else:
                    remain.append(pax)
            
            vehicle.onboard = remain
    
    def _compile_metrics(self, total_tacc: float, steps: int) -> Dict[str, Any]:
        """Compile episode metrics matching Stage 1 schema."""
        total_requests = float(len(self.env.requests))
        structural = float(self.env.structurally_unserviceable)
        waiting_churned = float(self.env.waiting_churned)
        waiting_timeouts = float(self.env.waiting_timeouts)
        onboard_churned = float(self.env.onboard_churned)
        served = float(self.env.served)
        
        non_structural = max(0.0, total_requests - structural)
        waiting_total = waiting_churned + waiting_timeouts
        algorithmic = waiting_total + onboard_churned
        
        service_rate = served / non_structural if non_structural > 0 else 0.0
        waiting_churn_rate = waiting_total / non_structural if non_structural > 0 else 0.0
        onboard_churn_rate = onboard_churned / non_structural if non_structural > 0 else 0.0
        algorithmic_churn_rate = algorithmic / non_structural if non_structural > 0 else 0.0
        structural_rate = structural / total_requests if total_requests > 0 else 0.0
        
        wait_times = []
        for req in self.env.requests:
            pickup = req.get("pickup_time_sec")
            if pickup is not None:
                wait = pickup - req["request_time_sec"]
                wait_times.append(max(0.0, wait))
        
        wait_p95 = float(np.percentile(wait_times, 95)) if wait_times else 0.0
        
        gini = self.env._gini([float(v) for v in self.env.service_count_by_stop.values()])
        
        sim_to_real = self.env.get_sim_to_real_summary()
        
        return {
            "total_requests": total_requests,
            "served": served,
            "waiting_churned": waiting_churned,
            "waiting_timeouts": waiting_timeouts,
            "onboard_churned": onboard_churned,
            "structural_unserviceable": structural,
            "service_rate": float(service_rate),
            "waiting_churn_rate": float(waiting_churn_rate),
            "onboard_churn_rate": float(onboard_churn_rate),
            "algorithmic_churn_rate": float(algorithmic_churn_rate),
            "structural_unserviceable_rate": float(structural_rate),
            "tacc_total": float(total_tacc),
            "wait_time_p95_sec": float(wait_p95),
            "service_gini": float(gini),
            "steps": float(steps),
            "sim_to_real": sim_to_real,
        }


def run_traci_validation(
    sumo_cfg_path: str,
    env_config: Any,
    policy_fn: Callable,
    stop_to_edge: Dict[int, str],
    max_steps: int = 1000,
    sumo_gui: bool = False,
    sumo_seed: int = 7,
) -> Dict[str, Any]:
    """Run policy validation in SUMO and collect metrics.
    
    Args:
        sumo_cfg_path: Path to SUMO configuration file
        env_config: SUMOEnvConfig instance
        policy_fn: Callable that takes features dict and returns action
        stop_to_edge: Mapping from logical stop IDs to SUMO edge IDs
        max_steps: Maximum simulation steps
        sumo_gui: Whether to use SUMO GUI
        sumo_seed: Random seed for SUMO
    
    Returns:
        Dict with episode metrics matching Stage 1 schema plus sim-to-real deltas
    """
    from .sumo_env import SUMOEnv
    
    traci_config = TraCIConfig(
        sumo_cfg_path=sumo_cfg_path,
        sumo_gui=sumo_gui,
        sumo_seed=sumo_seed,
    )
    
    env = SUMOEnv(env_config)
    
    runner = TraCISimulationRunner(
        traci_config=traci_config,
        sumo_env=env,
        policy_fn=policy_fn,
        stop_to_edge=stop_to_edge,
    )
    
    return runner.run_episode(max_steps=max_steps)
