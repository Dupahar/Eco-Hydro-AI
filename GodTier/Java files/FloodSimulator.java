import java.util.ArrayList;
import java.util.List;

/**
 * Physics-Informed Flood Simulator
 * Implements simplified shallow water equations with graph-based propagation
 * 
 * Based on Team Dupahar's DUALFloodGNN architecture:
 * - Mass conservation (continuity equation)
 * - Momentum conservation (Manning's equation approximation)
 * - Building impermeability (topological graph cuts)
 */
public class FloodSimulator {
    
    private DataManager dataManager;
    
    // Simulation state
    private double[][] waterDepths;      // Current water depth at each cell (meters)
    private double[][] velocityX;        // X-component of velocity (m/s)
    private double[][] velocityY;        // Y-component of velocity (m/s)
    private boolean[][] isBuilding;      // Building mask
    
    // Simulation parameters
    private double rainfallRate = 0.0;   // mm/hour
    private double infiltrationRate = 5.0; // mm/hour (soil absorption)
    private double manningCoefficient = 0.03; // Roughness coefficient
    private double timeStep = 0.1;       // seconds
    private double cellSize = 5.0;       // meters
    
    // Simulation control
    private boolean running = false;
    private double simulatedTime = 0.0;
    
    // Precomputed graph structure (for efficiency)
    private List<GraphEdge> flowGraph;
    
    public FloodSimulator(DataManager dataManager) {
        this.dataManager = dataManager;
    }
    
    /**
     * Initialize simulation with terrain data
     */
    public void initialize() {
        TerrainData terrain = dataManager.getTerrainData();
        BuildingData buildings = dataManager.getBuildingData();
        
        if (terrain == null) return;
        
        int rows = terrain.getRows();
        int cols = terrain.getCols();
        
        // Initialize arrays
        waterDepths = new double[rows][cols];
        velocityX = new double[rows][cols];
        velocityY = new double[rows][cols];
        isBuilding = new boolean[rows][cols];
        
        // Mark building cells
        if (buildings != null) {
            for (Building building : buildings.getBuildings()) {
                markBuildingCells(building, terrain);
            }
        }
        
        // Build flow graph (connectivity between cells)
        buildFlowGraph(terrain);
        
        simulatedTime = 0.0;
    }
    
    /**
     * Mark cells occupied by buildings
     */
    private void markBuildingCells(Building building, TerrainData terrain) {
        double[][] vertices = building.getVertices();
        
        // Simple rasterization: mark all cells inside building polygon
        int minRow = Integer.MAX_VALUE, maxRow = Integer.MIN_VALUE;
        int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
        
        for (double[] vertex : vertices) {
            int col = (int)(vertex[0] / cellSize);
            int row = (int)(vertex[1] / cellSize);
            minCol = Math.min(minCol, col);
            maxCol = Math.max(maxCol, col);
            minRow = Math.min(minRow, row);
            maxRow = Math.max(maxRow, row);
        }
        
        for (int r = minRow; r <= maxRow && r < isBuilding.length; r++) {
            for (int c = minCol; c <= maxCol && c < isBuilding[0].length; c++) {
                if (r >= 0 && c >= 0 && isPointInPolygon(c * cellSize, r * cellSize, vertices)) {
                    isBuilding[r][c] = true;
                }
            }
        }
    }
    
    /**
     * Check if point is inside polygon (ray casting algorithm)
     */
    private boolean isPointInPolygon(double x, double y, double[][] polygon) {
        int intersections = 0;
        int n = polygon.length;
        
        for (int i = 0; i < n; i++) {
            double x1 = polygon[i][0];
            double y1 = polygon[i][1];
            double x2 = polygon[(i + 1) % n][0];
            double y2 = polygon[(i + 1) % n][1];
            
            if (((y1 > y) != (y2 > y)) && 
                (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)) {
                intersections++;
            }
        }
        
        return (intersections % 2) == 1;
    }
    
    /**
     * Build flow graph representing water connectivity
     */
    private void buildFlowGraph(TerrainData terrain) {
        flowGraph = new ArrayList<>();
        
        int rows = waterDepths.length;
        int cols = waterDepths[0].length;
        
        // Connect each cell to its 8 neighbors (if not blocked by buildings)
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (isBuilding[r][c]) continue; // Skip building cells
                
                // Check 8 neighbors
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue;
                        
                        int nr = r + dr;
                        int nc = c + dc;
                        
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !isBuilding[nr][nc]) {
                            double distance = Math.sqrt(dr * dr + dc * dc) * cellSize;
                            flowGraph.add(new GraphEdge(r, c, nr, nc, distance));
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Perform one simulation time step
     */
    public void step(double deltaTime) {
        if (!running) return;
        
        TerrainData terrain = dataManager.getTerrainData();
        if (terrain == null) return;
        
        int rows = waterDepths.length;
        int cols = waterDepths[0].length;
        
        // 1. Add rainfall
        applyRainfall(deltaTime);
        
        // 2. Compute water surface elevations
        double[][] waterSurface = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                waterSurface[r][c] = terrain.getElevations()[r][c] + waterDepths[r][c];
            }
        }
        
        // 3. Propagate water (physics-informed diffusion)
        double[][] newDepths = new double[rows][cols];
        double[][] newVelX = new double[rows][cols];
        double[][] newVelY = new double[rows][cols];
        
        // Copy current depths
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                newDepths[r][c] = waterDepths[r][c];
            }
        }
        
        // Process each flow edge
        for (GraphEdge edge : flowGraph) {
            int r1 = edge.fromRow, c1 = edge.fromCol;
            int r2 = edge.toRow, c2 = edge.toCol;
            
            // Hydraulic gradient (difference in water surface elevation)
            double dh = waterSurface[r1][c1] - waterSurface[r2][c2];
            
            if (Math.abs(dh) < 0.001) continue; // No gradient, no flow
            
            // Average water depth along edge
            double avgDepth = (waterDepths[r1][c1] + waterDepths[r2][c2]) / 2.0;
            
            if (avgDepth < 0.01) continue; // Too shallow to flow
            
            // Manning's equation approximation for flow velocity
            // v = (1/n) * R^(2/3) * S^(1/2)
            // where R ≈ depth, S = slope, n = Manning coefficient
            double slope = Math.abs(dh) / edge.distance;
            double velocity = (1.0 / manningCoefficient) * 
                             Math.pow(avgDepth, 2.0/3.0) * 
                             Math.sqrt(slope);
            
            // Flow rate (m³/s per meter width)
            double flowRate = velocity * avgDepth;
            
            // Volume transferred in this timestep
            double volume = flowRate * deltaTime;
            
            // Limit transfer to prevent negative depths
            double maxTransfer = Math.min(volume, waterDepths[r1][c1] * cellSize * cellSize * 0.5);
            
            if (dh > 0) {
                // Flow from cell 1 to cell 2
                newDepths[r1][c1] -= maxTransfer / (cellSize * cellSize);
                newDepths[r2][c2] += maxTransfer / (cellSize * cellSize);
                
                // Update velocities
                double dx = (c2 - c1) * cellSize / edge.distance;
                double dy = (r2 - r1) * cellSize / edge.distance;
                newVelX[r1][c1] += velocity * dx;
                newVelY[r1][c1] += velocity * dy;
            } else {
                // Flow from cell 2 to cell 1
                newDepths[r2][c2] -= maxTransfer / (cellSize * cellSize);
                newDepths[r1][c1] += maxTransfer / (cellSize * cellSize);
                
                double dx = (c1 - c2) * cellSize / edge.distance;
                double dy = (r1 - r2) * cellSize / edge.distance;
                newVelX[r2][c2] += velocity * dx;
                newVelY[r2][c2] += velocity * dy;
            }
        }
        
        // 4. Apply infiltration (soil absorption)
        applyInfiltration(newDepths, deltaTime);
        
        // 5. Update state
        waterDepths = newDepths;
        
        // Normalize velocities
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                velocityX[r][c] = newVelX[r][c] / 8.0; // Average over 8 neighbors
                velocityY[r][c] = newVelY[r][c] / 8.0;
            }
        }
        
        simulatedTime += deltaTime;
    }
    
    /**
     * Add rainfall to all non-building cells
     */
    private void applyRainfall(double deltaTime) {
        if (rainfallRate <= 0) return;
        
        // Convert mm/hour to m/second
        double rainfallDepthPerSecond = (rainfallRate / 1000.0) / 3600.0;
        double addedDepth = rainfallDepthPerSecond * deltaTime;
        
        for (int r = 0; r < waterDepths.length; r++) {
            for (int c = 0; c < waterDepths[0].length; c++) {
                if (!isBuilding[r][c]) {
                    waterDepths[r][c] += addedDepth;
                }
            }
        }
    }
    
    /**
     * Remove water through soil infiltration
     */
    private void applyInfiltration(double[][] depths, double deltaTime) {
        double infiltrationDepthPerSecond = (infiltrationRate / 1000.0) / 3600.0;
        double removedDepth = infiltrationDepthPerSecond * deltaTime;
        
        for (int r = 0; r < depths.length; r++) {
            for (int c = 0; c < depths[0].length; c++) {
                if (!isBuilding[r][c]) {
                    depths[r][c] = Math.max(0, depths[r][c] - removedDepth);
                }
            }
        }
    }
    
    /**
     * Set rainfall scenario
     */
    public void setRainfallScenario(RainfallScenario scenario) {
        switch (scenario) {
            case LIGHT:
                rainfallRate = 10.0; // 10 mm/hour
                break;
            case MODERATE:
                rainfallRate = 30.0; // 30 mm/hour
                break;
            case HEAVY:
                rainfallRate = 75.0; // 75 mm/hour
                break;
            case EXTREME:
                rainfallRate = 150.0; // 150 mm/hour
                break;
            case CUSTOM:
                // rainfallRate set manually
                break;
        }
    }
    
    public enum RainfallScenario {
        LIGHT, MODERATE, HEAVY, EXTREME, CUSTOM
    }
    
    /**
     * Reset simulation to initial state
     */
    public void reset() {
        initialize();
        simulatedTime = 0.0;
    }
    
    /**
     * Start/stop simulation
     */
    public void start() { 
        if (waterDepths == null) initialize();
        running = true; 
    }
    
    public void stop() { running = false; }
    public boolean isRunning() { return running; }
    
    /**
     * Getters for visualization
     */
    public double[][] getCurrentFloodDepths() { return waterDepths; }
    
    public VelocityField getCurrentVelocities() {
        if (velocityX == null || velocityY == null) return null;
        return new VelocityField(velocityX, velocityY);
    }
    
    public double getSimulatedTime() { return simulatedTime; }
    
    // Setters
    public void setRainfallRate(double rate) { this.rainfallRate = rate; }
    public void setInfiltrationRate(double rate) { this.infiltrationRate = rate; }
    public void setManningCoefficient(double n) { this.manningCoefficient = n; }
    
    /**
     * Inner class representing a graph edge for water flow
     */
    private static class GraphEdge {
        int fromRow, fromCol;
        int toRow, toCol;
        double distance;
        
        GraphEdge(int fromRow, int fromCol, int toRow, int toCol, double distance) {
            this.fromRow = fromRow;
            this.fromCol = fromCol;
            this.toRow = toRow;
            this.toCol = toCol;
            this.distance = distance;
        }
    }
}
