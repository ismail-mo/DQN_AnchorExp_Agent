import gymnasium as gym
from gymnasium import spaces
import pygame
from Box2D import (
    b2World, b2PolygonShape, b2FixtureDef, b2CircleShape,
    b2PrismaticJointDef, b2ContactListener, b2ContactImpulse
)
import time
import cv2  # Add OpenCV for video recording
import os
import numpy as np
import Box2D

# ================== Environment Constants ==================
CHAMBER_WIDTH = 0.8  # meters
CHAMBER_HEIGHT = 1.0  # meters
MARGIN = 0.1          # meters border
SCALE = 400.0         # px per meter
VIEWPORT_W = int((CHAMBER_WIDTH + 2 * MARGIN) * SCALE)
VIEWPORT_H = int((CHAMBER_HEIGHT + 2 * MARGIN) * SCALE)
FPS = 60
TIME_STEP = 1.0 / FPS
VEL_ITERS, POS_ITERS = 10, 10

# Grain parameters
GRAIN_RADIUS = 0.012  # m

# ================== World Setup ==================
def create_grains(world):
    grains = []
    
    # Define the grain arrangement first
    grain_diameter = GRAIN_RADIUS * 2
    
    # Calculate rows and columns based on chamber width, not including walls yet
    COLS = int(CHAMBER_WIDTH / grain_diameter) + 2  # Add extra columns to ensure full coverage
    
    # Fill 1/3 of chamber height with grains
    soil_height = CHAMBER_HEIGHT / 3
    ROWS = int(soil_height / (grain_diameter * 0.84))  # Reduced from 0.866 for tighter packing
    
    # Center the array horizontally
    x0 = -(COLS-1) * grain_diameter / 2
    
    # Position grains starting at y=0 (we'll adjust the chamber floor later)
    y0 = GRAIN_RADIUS  # Bottom of first grain at y=0
    
    for i in range(ROWS):
        for j in range(COLS):
            # Offset alternate rows for hexagonal packing
            x = x0 + j * grain_diameter + (0.5 * grain_diameter if i % 2 else 0)
            y = y0 + i * (grain_diameter * 0.84)  # Reduced from 0.866 for tighter packing
            
            # Add random jitter to avoid perfect lattice arrangement
            x += np.random.uniform(-0.002, 0.002)
            y += np.random.uniform(-0.002, 0.002)
            
            # Allow grains to extend closer to chamber walls (only skip if majority of grain would be outside)
            if abs(x) > (CHAMBER_WIDTH/2 + GRAIN_RADIUS/2):
                continue
                
            body = world.CreateDynamicBody(
                position=(x, y),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=GRAIN_RADIUS),
                    density=2.5,       # Increased mass to resist movement
                    friction=0.6,      # friction for resistance
                    restitution=0.0    # no bounce
                ),
                linearDamping=0.5,   # Reduced from 0.8 to allow more flow
                angularDamping=0.6    # Reduced from 0.9 to allow more rotation
            )
            # Enable continuous collision detection to prevent tunneling
            body.bullet = True
            grains.append(body)
    
    return grains

def create_chamber(world):
    # Wall thickness
    t = 0.01
    
    # Calculate half width
    half_w = CHAMBER_WIDTH / 2
    
    walls = []
    
    # Left wall - place it just outside the leftmost grains
    walls.append(world.CreateStaticBody(
        position=(-half_w - t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
    ))
    
    # Right wall - place it just outside the rightmost grains
    walls.append(world.CreateStaticBody(
        position=(half_w + t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
    ))
    
    # Floor - place it directly beneath the grains at y=0
    walls.append(world.CreateStaticBody(
        position=(0.0, -t/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w + t, t/2)), friction=0.6)
    ))
    
    return walls

# ================== Probe Parameters ==================
# Probe parameters
SHAFT_W = 0.075  # Increased from 0.05
SHAFT_H = 0.5
TIP_BASE = SHAFT_W
TIP_HEIGHT = SHAFT_W * 0.6

# Position the probe closer to the soil bed
soil_height = CHAMBER_HEIGHT / 3
PROBE_POS_Y = soil_height + 0.05  # Small clearance above soil
EXPANSION_RATIO = 1.2
EXPANSION_DISTANCE = 0.05  # Target expansion distance (meters)

# Speed parameters
NORMAL_SPEED = 0.05   # m/s for expansion
MAX_SAFE_SPEED = 0.1  # m/s maximum safe speed
MAX_FORCE = 50.0      # N maximum expected force
MAX_STEPS = 500       # Maximum episode length

# ================== Contact Listener ==================
class ContactDetector(b2ContactListener):
    def __init__(self, probe_parts):
        super().__init__()
        self.probe_parts = probe_parts
        self.probe_impulse = 0.0

    def PostSolve(self, contact, impulse: b2ContactImpulse):
        total = sum(impulse.normalImpulses)
        bodies = (contact.fixtureA.body, contact.fixtureB.body)
        if any(part in bodies for part in self.probe_parts):
            self.probe_impulse += total

# ================== Helper Functions ==================
def world_to_screen(x, y):
    # Move the origin to bottom-left corner of the chamber
    sx = int((x + CHAMBER_WIDTH/2 + MARGIN) * SCALE)
    
    # Flip Y-axis and position the origin at the bottom with a small margin
    bottom_margin = MARGIN * 2  # Extra margin at the bottom
    sy = int(VIEWPORT_H - ((y + bottom_margin) * SCALE))
    
    return sx, sy

# ================== AnchorExp Environment ==================
class AnchorExpEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None):
        # Initialize rendering components
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_mode = render_mode
        
        # Initialize physics world
        self.world = None
        self.chamber = None
        self.grains = []
        self.left_shaft = None
        self.right_shaft = None
        self.tip = None
        self.anchor_joint = None
        
        # State tracking
        self.current_expansion = 0.0
        self.current_velocity = 0.0
        self.current_force = 0.0
        self.initial_distance = 0.0
        self.step_count = 0
        self.contact_detector = None
        
        # Set max episode steps directly in the environment
        self.max_episode_steps = MAX_STEPS  # Use the constant defined at module level
        
        # Define observation space (normalized values)
        # [expansion_ratio, velocity, force, time_step_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space
        # 0: Expand, 1: Contract, 2: Stop
        self.action_space = spaces.Discrete(3)
    
    def _destroy(self):
        """Clean up Box2D objects"""
        if self.world is None:
            return
            
        # Destroy grains
        for grain in self.grains:
            self.world.DestroyBody(grain)
        self.grains = []
        
        # Destroy probe parts
        if self.left_shaft is not None:
            self.world.DestroyBody(self.left_shaft)
            self.left_shaft = None
        if self.right_shaft is not None:
            self.world.DestroyBody(self.right_shaft)
            self.right_shaft = None
        if self.tip is not None:
            self.world.DestroyBody(self.tip)
            self.tip = None
        
        # Destroy chamber
        if self.chamber is not None:
            for wall in self.chamber:
                self.world.DestroyBody(wall)
            self.chamber = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._destroy()
        
        # Create new world with downward gravity
        self.world = b2World(gravity=(0.0, -5.0))
        
        # Create chamber
        self.chamber = self._create_chamber()
        
        # Create the split probe (left shaft, right shaft, and tip)
        self.left_shaft, self.right_shaft, self.tip = self._create_probe()
        
        # Get probe positions and dimensions for collision avoidance
        probe_x_min = min(self.left_shaft.position.x - SHAFT_W/4, self.tip.position.x - TIP_BASE/2)
        probe_x_max = max(self.right_shaft.position.x + SHAFT_W/4, self.tip.position.x + TIP_BASE/2)
        probe_y_min = self.tip.position.y - TIP_HEIGHT/2
        probe_y_max = max(self.left_shaft.position.y + SHAFT_H/2, self.right_shaft.position.y + SHAFT_H/2)
        
        # Create grains avoiding the probe area
        self.grains = self._create_grains(probe_x_min, probe_x_max, probe_y_min, probe_y_max)
        
        # Create anchor joint between shafts
        self.anchor_joint = self._create_anchor_joint(self.left_shaft, self.right_shaft)
        
        # Create contact detector
        self.contact_detector = ContactDetector([self.left_shaft, self.right_shaft, self.tip])
        self.world.contactListener = self.contact_detector
        
        # Reset state variables
        self.step_count = 0
        self.initial_distance = abs(self.right_shaft.position.x - self.left_shaft.position.x)
        self.current_expansion = 0.0
        self.current_velocity = 0.0
        self.current_force = 0.0
        
        # Let the world settle briefly
        for _ in range(10):
            self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        
        # Update observation
        self._update_state()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self.step_count += 1
        
        # Reset force measurement before physics step
        self.contact_detector.probe_impulse = 0.0
        
        # Apply action
        if action == 0:  # Expand
            self.anchor_joint.motorSpeed = NORMAL_SPEED
        elif action == 1:  # Contract
            self.anchor_joint.motorSpeed = -NORMAL_SPEED
        elif action == 2:  # Stop
            self.anchor_joint.motorSpeed = 0.0
        
        # Step physics with interruption protection
        try:
            # Run physics step with keyboard interrupt protection
            self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        except KeyboardInterrupt:
            print("\nSafely handling keyboard interrupt during physics step...")
            # Clean up resources
            self.close()
            raise  # Re-raise the interrupt to allow proper program exit
        
        # Update state variables
        self._update_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Success: reached target expansion (95-105% of target)
        if 0.95 <= self.current_expansion <= 1.05:
            terminated = True
            reward += 10.0  # Completion bonus
        
        # Failure: overexpanded
        elif self.current_expansion > 1.1:
            terminated = True
            reward -= 10.0  # Over-expansion penalty
        
        # Truncation: exceeded maximum steps
        if self.step_count >= MAX_STEPS:
            truncated = True
            reward -= 5.0  # Timeout penalty
        
        if self.render_mode == "human":
            self.render()
        
        # Get the observations
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}
    
    def _update_state(self):
        """Update the state variables based on current simulation state"""
        # Calculate the current expansion as a ratio of the target expansion
        distance = abs(self.right_shaft.position.x - self.left_shaft.position.x)
        target_distance = self.initial_distance + EXPANSION_DISTANCE
        self.current_expansion = (distance - self.initial_distance) / EXPANSION_DISTANCE
        
        # Get current velocity (normalized)
        self.current_velocity = self.anchor_joint.speed / MAX_SAFE_SPEED
        
        # Get current force - don't normalize here, we'll clip in _get_observation
        self.current_force = self.contact_detector.probe_impulse
    
    def _calculate_reward(self):
        """Calculate the reward for the current state"""
        reward = 0.0
        
        # 1. Progress reward: reward for progress toward target expansion
        prev_expansion = self.current_expansion - self.anchor_joint.speed * TIME_STEP / EXPANSION_DISTANCE
        expansion_progress = self.current_expansion - prev_expansion
        reward += expansion_progress * 5.0  # Scale up the progress reward
        
        # 2. Efficiency penalty: small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # 3. Force penalty: penalty proportional to the force on the anchor
        force_penalty = 0.1 * self.current_force
        reward -= force_penalty
        
        # 4 & 5. Completion bonus and over-expansion penalty are handled in the step function
        # where we can terminate the episode
        
        return reward
    
    def _get_observation(self):
        """Get the current observation vector"""
        # 1. Expansion ratio: non-negative, typically 0-2 range
        expansion_ratio = np.clip(self.current_expansion, 0.0, 2.0)
        
        # 2. Joint speed: normalized to [-1, +1] range
        velocity = np.clip(self.current_velocity, -1.0, 1.0)
        
        # 3. Contact force: normalized to [0, 1] range (not [-1, 1])
        # Take absolute value if needed and normalize to [0, 1]
        force = np.clip(abs(self.current_force) / MAX_FORCE, 0.0, 1.0)
        
        # 4. Time fraction: [0, 1] range
        time_fraction = np.clip(self.step_count / self.max_episode_steps, 0.0, 1.0)
        
        # Ensure array data type is float32 to match observation space
        observation = np.array([expansion_ratio, velocity, force, time_fraction], dtype=np.float32)
        
        # Final verification against observation space
        low = self.observation_space.low
        high = self.observation_space.high
        observation = np.maximum(np.minimum(observation, high), low)
        
        return observation
    
    def _create_chamber(self):
        """Create the chamber walls"""
        # Wall thickness
        t = 0.01
        
        # Calculate half width
        half_w = CHAMBER_WIDTH / 2
        
        walls = []
        
        # Left wall - place it just outside the leftmost grains
        walls.append(self.world.CreateStaticBody(
            position=(-half_w - t/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
        ))
        
        # Right wall - place it just outside the rightmost grains
        walls.append(self.world.CreateStaticBody(
            position=(half_w + t/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
        ))
        
        # Floor - place it directly beneath the grains at y=0
        walls.append(self.world.CreateStaticBody(
            position=(0.0, -t/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w + t, t/2)), friction=0.6)
        ))
        
        return walls
    
    def _create_grains(self, probe_x_min, probe_x_max, probe_y_min, probe_y_max):
        """Create soil grains while avoiding the probe area"""
        grains = []
        
        # Define the grain arrangement first
        grain_diameter = GRAIN_RADIUS * 2
        
        # Calculate rows and columns based on chamber width
        COLS = int(CHAMBER_WIDTH / grain_diameter) + 2  # Add extra columns
        
        # Fill 1/3 of chamber height with grains
        soil_height = CHAMBER_HEIGHT / 3
        ROWS = int(soil_height / (grain_diameter * 0.84))  # Reduced from 0.866 for tighter packing
        
        # Center the array horizontally
        x0 = -(COLS-1) * grain_diameter / 2
        
        # Position grains starting at y=0 (we'll adjust the chamber floor later)
        y0 = GRAIN_RADIUS  # Bottom of first grain at y=0
        
        # Calculate tip width at different heights for more precise collision detection
        # The tip is triangular, so its width varies with height
        def get_tip_width_at_height(height_from_tip_bottom):
            # At tip_bottom (height=0), width is 0
            # At tip_top (height=TIP_HEIGHT), width is TIP_BASE
            if height_from_tip_bottom <= 0:
                return 0
            elif height_from_tip_bottom >= TIP_HEIGHT:
                return TIP_BASE
            else:
                return (height_from_tip_bottom / TIP_HEIGHT) * TIP_BASE
        
        for i in range(ROWS):
            for j in range(COLS):
                # Offset alternate rows for hexagonal packing
                x = x0 + j * grain_diameter + (0.5 * grain_diameter if i % 2 else 0)
                y = y0 + i * (grain_diameter * 0.84)  # Reduced from 0.866 for tighter packing
                
                # Add random jitter to avoid perfect lattice arrangement
                x += np.random.uniform(-0.002, 0.002)
                y += np.random.uniform(-0.002, 0.002)
                
                # Skip if grain would be outside chamber
                if abs(x) > (CHAMBER_WIDTH/2 + GRAIN_RADIUS/2):
                    continue
                
                # Probe collision check with special handling for the tip area
                buffer = GRAIN_RADIUS * 1.1
                
                # Check if we're in the tip area
                if y < probe_y_min + TIP_HEIGHT and y >= probe_y_min - buffer:
                    # Calculate how far we are from the tip bottom
                    height_from_tip_bottom = y - (probe_y_min - TIP_HEIGHT/2) + buffer
                    # Get width of tip at this height
                    tip_width = get_tip_width_at_height(height_from_tip_bottom)
                    # Only skip if we're within the actual triangular shape + buffer
                    tip_x_min = -tip_width/2
                    tip_x_max = tip_width/2
                    if abs(x) < tip_width/2 + buffer:
                        continue
                # For non-tip areas, use standard bounding box check
                elif (x + buffer > probe_x_min and x - buffer < probe_x_max and 
                      y + buffer > probe_y_min and y - buffer < probe_y_max):
                    continue
                    
                body = self.world.CreateDynamicBody(
                    position=(x, y),
                    fixtures=b2FixtureDef(
                        shape=b2CircleShape(radius=GRAIN_RADIUS),
                        density=2.5,       # Increased mass to resist movement
                        friction=0.6,      # friction for resistance
                        restitution=0.0    # no bounce
                    ),
                    linearDamping=0.5,   # Reduced from 0.8 to allow more flow
                    angularDamping=0.6    # Reduced from 0.9 to allow more rotation
                )
                # Enable continuous collision detection to prevent tunneling
                body.bullet = True
                grains.append(body)
        
        return grains
    
    def _create_probe(self):
        """Create split probe (left shaft, right shaft, and tip)"""
        # Create left shaft
        left_shaft = self.world.CreateDynamicBody(
            position=(-SHAFT_W/4, PROBE_POS_Y),
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(SHAFT_W/4, SHAFT_H/2)),
                density=1.0, friction=0.2
            )
        )
        
        # Create right shaft
        right_shaft = self.world.CreateDynamicBody(
            position=(SHAFT_W/4, PROBE_POS_Y),
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(SHAFT_W/4, SHAFT_H/2)),
                density=1.0, friction=0.2
            )
        )
        
        # Create tip - position it at the bottom of shafts
        tip = self.world.CreateDynamicBody(
            position=(0.0, PROBE_POS_Y - SHAFT_H/2 - TIP_HEIGHT/2),
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[
                    (-TIP_BASE/2, TIP_HEIGHT/2),
                    (TIP_BASE/2, TIP_HEIGHT/2),
                    (0.0, -TIP_HEIGHT/2)
                ]),
                density=1.0, friction=0.2
            )
        )
        
        # Disable gravity for all probe parts
        left_shaft.gravityScale = 0.0
        right_shaft.gravityScale = 0.0
        tip.gravityScale = 0.0
        
        # Create weld joints to connect tip to both shaft halves
        self.world.CreateWeldJoint(
            bodyA=left_shaft,
            bodyB=tip,
            anchor=(left_shaft.position.x, tip.position.y + TIP_HEIGHT/2),
            referenceAngle=0.0
        )
        
        self.world.CreateWeldJoint(
            bodyA=right_shaft,
            bodyB=tip,
            anchor=(right_shaft.position.x, tip.position.y + TIP_HEIGHT/2),
            referenceAngle=0.0
        )
        
        return left_shaft, right_shaft, tip
    
    def _create_anchor_joint(self, left, right):
        """Create a prismatic joint between left and right shafts"""
        pj = b2PrismaticJointDef()
        pj.Initialize(left, right, anchor=(0, left.position.y), axis=(1, 0))
        
        pj.enableLimit = True
        pj.lowerTranslation = 0.0  # Cannot get closer
        pj.upperTranslation = SHAFT_W * EXPANSION_RATIO  # Max expansion distance
        
        pj.enableMotor = True
        pj.motorSpeed = 0.0  # Start with no movement
        pj.maxMotorForce = 500.0
        
        return self.world.CreateJoint(pj)
    
    def _create_anchor(self):
        # Define anchor vertices for a triangular shape
        vertices = [
            (-0.5, -0.5),  # bottom left
            (0.5, -0.5),   # bottom right
            (0.0, 0.5)     # top center
        ]
        
        # Create anchor body
        anchor_body = self.world.CreateDynamicBody(
            position=(self.ANCHOR_X, self.ANCHOR_Y),
            fixtures=b2PolygonShape(vertices=vertices),
            userData='anchor'
        )
        
        return anchor_body

    # For more complex shapes, you can use multiple fixtures
    def _create_complex_anchor(self):
        # Create the main body
        anchor_body = self.world.CreateDynamicBody(
            position=(self.ANCHOR_X, self.ANCHOR_Y),
            userData='anchor'
        )
        
        # Add multiple polygon fixtures to create complex shape
        # Main body
        body_vertices = [
            (-0.3, -0.5),
            (0.3, -0.5),
            (0.3, 0.3),
            (-0.3, 0.3)
        ]
        anchor_body.CreateFixture(
            shape=b2PolygonShape(vertices=body_vertices),
            density=1.0,
            friction=0.3
        )
        
        # Top point
        tip_vertices = [
            (-0.2, 0.3),
            (0.2, 0.3),
            (0.0, 0.7)
        ]
        anchor_body.CreateFixture(
            shape=b2PolygonShape(vertices=tip_vertices),
            density=0.5,
            friction=0.3
        )
        
        return anchor_body
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
            
        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Anchor Expansion")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.screen.fill((255,255,255))  # White background
        
        # Draw chamber walls
        for body in self.chamber:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    vertices = [body.transform * v for v in shape.vertices]
                    points = [world_to_screen(v[0], v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, (50, 50, 50), points)
        
        # Draw grains
        for grain in self.grains:
            x, y = world_to_screen(grain.position.x, grain.position.y)
            pygame.draw.circle(self.screen, (150, 150, 150), (x, y), int(GRAIN_RADIUS * SCALE))
        
        # Draw left shaft
        for fixture in self.left_shaft.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2PolygonShape):
                vertices = [self.left_shaft.transform * v for v in shape.vertices]
                points = [world_to_screen(v[0], v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (100, 100, 255), points)
        
        # Draw right shaft
        for fixture in self.right_shaft.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2PolygonShape):
                vertices = [self.right_shaft.transform * v for v in shape.vertices]
                points = [world_to_screen(v[0], v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (100, 100, 255), points)
        
        # Draw tip
        for fixture in self.tip.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2PolygonShape):
                vertices = [self.tip.transform * v for v in shape.vertices]
                points = [world_to_screen(v[0], v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (255, 100, 100), points)
        
        # Display information
        text = self.font.render(f"Expansion: {self.current_expansion:.3f}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        velocity_text = self.font.render(f"Speed: {self.current_velocity:.3f}", True, (0, 0, 0))
        self.screen.blit(velocity_text, (10, 40))
        
        force_text = self.font.render(f"Force: {self.current_force:.3f}", True, (0, 0, 0))
        self.screen.blit(force_text, (10, 70))
        
        step_text = self.font.render(f"Step: {self.step_count}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 100))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

def main():
    """Run the environment for debugging purposes"""
    env = AnchorExpEnv(render_mode="human")
    observation, info = env.reset()
    
    print("AnchorExp Environment Initialized")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Try different action patterns
    total_reward = 0
    
    try:
        running = True
        while running:
            # Simple policy:
            # - Expand if expansion < 0.9
            # - Contract if expansion > 1.05
            # - Stop if expansion between 0.9 and 1.05
            expansion = observation[0]
            if expansion < 0.9:
                action = 0  # Expand
            elif expansion > 1.05:
                action = 1  # Contract
            else:
                action = 2  # Stop
            
            # Take step with current action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Print debug info every 50 steps
            if env.step_count % 50 == 0:
                print(f"Step {env.step_count}: Expansion={observation[0]:.3f}, Vel={observation[1]:.3f}, Force={observation[2]:.3f}")
                print(f"Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}")
            
            # Check for keyboard interrupt
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode done: {'Success!' if observation[0] >= 1.0 and observation[0] <= 1.1 else 'Failed.'}")
                print(f"Final expansion: {observation[0]:.3f}, Total reward: {total_reward:.3f}")
                break
            
            # Cap the debug loop framerate
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        print("Manual interrupt")
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()




