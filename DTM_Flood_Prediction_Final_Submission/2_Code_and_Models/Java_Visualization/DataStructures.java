/**
 * Data structures for Eco-Hydro-AI Platform
 */

/**
 * Terrain elevation data
 */
class TerrainData {
    private double[][] elevations;
    private int rows;
    private int cols;
    private double minElevation;
    private double maxElevation;
    
    public TerrainData(double[][] elevations) {
        this.elevations = elevations;
        this.rows = elevations.length;
        this.cols = elevations[0].length;
        
        computeElevationRange();
    }
    
    private void computeElevationRange() {
        minElevation = Double.MAX_VALUE;
        maxElevation = Double.MIN_VALUE;
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                minElevation = Math.min(minElevation, elevations[r][c]);
                maxElevation = Math.max(maxElevation, elevations[r][c]);
            }
        }
    }
    
    public double[][] getElevations() { return elevations; }
    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double getMinElevation() { return minElevation; }
    public double getMaxElevation() { return maxElevation; }
    
    public double getElevationAt(double x, double y) {
        // Bilinear interpolation
        int c = (int)(x / 5.0);
        int r = (int)(y / 5.0);
        
        if (c < 0 || c >= cols - 1 || r < 0 || r >= rows - 1) {
            return 0.0;
        }
        
        double fx = (x / 5.0) - c;
        double fy = (y / 5.0) - r;
        
        double v00 = elevations[r][c];
        double v10 = elevations[r][c + 1];
        double v01 = elevations[r + 1][c];
        double v11 = elevations[r + 1][c + 1];
        
        return (1 - fx) * (1 - fy) * v00 +
               fx * (1 - fy) * v10 +
               (1 - fx) * fy * v01 +
               fx * fy * v11;
    }
}

/**
 * Building footprint data
 */
class BuildingData {
    private java.util.List<Building> buildings;
    
    public BuildingData() {
        buildings = new java.util.ArrayList<>();
    }
    
    public void addBuilding(Building building) {
        buildings.add(building);
    }
    
    public java.util.List<Building> getBuildings() {
        return buildings;
    }
}

/**
 * Individual building polygon
 */
class Building {
    private double[][] vertices; // [n][2] array of (x, y) coordinates
    private String id;
    private String type;
    
    public Building(String id, double[][] vertices) {
        this.id = id;
        this.vertices = vertices;
        this.type = "residential";
    }
    
    public Building(String id, double[][] vertices, String type) {
        this.id = id;
        this.vertices = vertices;
        this.type = type;
    }
    
    public double[][] getVertices() { return vertices; }
    public String getId() { return id; }
    public String getType() { return type; }
}

/**
 * Velocity field data
 */
class VelocityField {
    private double[][] vx;
    private double[][] vy;
    
    public VelocityField(double[][] vx, double[][] vy) {
        this.vx = vx;
        this.vy = vy;
    }
    
    public int getRows() { return vx.length; }
    public int getCols() { return vx[0].length; }
    
    public double getVx(int row, int col) { return vx[row][col]; }
    public double getVy(int row, int col) { return vy[row][col]; }
    
    public double getMagnitude(int row, int col) {
        return Math.sqrt(vx[row][col] * vx[row][col] + vy[row][col] * vy[row][col]);
    }
}

/**
 * Data Manager - handles loading and caching of village data
 */
class DataManager {
    private TerrainData terrainData;
    private BuildingData buildingData;
    private String currentVillage;
    
    public void loadVillageData(String dtmPath, String buildingsPath, String simulationPath) {
        // In a real implementation, this would load from files
        // For demo purposes, generate synthetic data
        terrainData = generateSyntheticTerrain();
        buildingData = generateSyntheticBuildings();
        currentVillage = dtmPath;
    }
    
    /**
     * Generate synthetic terrain for demonstration
     */
    private TerrainData generateSyntheticTerrain() {
        int rows = 150;
        int cols = 150;
        double[][] elevations = new double[rows][cols];
        
        // Create a realistic terrain with natural drainage patterns
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                // Base elevation with gradient
                double baseElevation = 10.0 + (rows - r) * 0.1 + (cols - c) * 0.05;
                
                // Add small-scale variation
                double noise = Math.sin(r * 0.2) * Math.cos(c * 0.15) * 2.0;
                
                // Add some natural depressions (potential flooding areas)
                double depression1 = -3.0 * Math.exp(-((r - 50) * (r - 50) + (c - 50) * (c - 50)) / 200.0);
                double depression2 = -2.5 * Math.exp(-((r - 100) * (r - 100) + (c - 100) * (c - 100)) / 300.0);
                
                elevations[r][c] = baseElevation + noise + depression1 + depression2;
            }
        }
        
        return new TerrainData(elevations);
    }
    
    /**
     * Generate synthetic building footprints
     */
    private BuildingData generateSyntheticBuildings() {
        BuildingData buildings = new BuildingData();
        
        // Create some typical Abadi-style buildings (small rectangular structures)
        java.util.Random rand = new java.util.Random(42);
        
        for (int i = 0; i < 30; i++) {
            double x = 100 + rand.nextDouble() * 500;
            double y = 100 + rand.nextDouble() * 500;
            double width = 15 + rand.nextDouble() * 20;
            double height = 12 + rand.nextDouble() * 18;
            
            double[][] vertices = {
                {x, y},
                {x + width, y},
                {x + width, y + height},
                {x, y + height}
            };
            
            buildings.addBuilding(new Building("B" + i, vertices, "residential"));
        }
        
        // Add a few larger buildings (community structures)
        double[][] school = {
            {400, 300}, {450, 300}, {450, 340}, {400, 340}
        };
        buildings.addBuilding(new Building("school", school, "public"));
        
        double[][] temple = {
            {250, 450}, {280, 450}, {280, 480}, {250, 480}
        };
        buildings.addBuilding(new Building("temple", temple, "religious"));
        
        return buildings;
    }
    
    public TerrainData getTerrainData() { return terrainData; }
    public BuildingData getBuildingData() { return buildingData; }
    public String getCurrentVillage() { return currentVillage; }
}
