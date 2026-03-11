import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.paint.LinearGradient;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.Stop;
import javafx.animation.AnimationTimer;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.geometry.Point3D;

/**
 * Advanced 3D Terrain Renderer with Real-time Flood Overlay
 * 
 * Capabilities:
 * - Isometric 3D projection of terrain
 * - Real-time flood depth heatmap overlay
 * - Interactive camera controls (pan, zoom, rotate)
 * - Building boundary visualization
 * - Traffic-light accessibility zones
 */
public class TerrainRenderer {
    
    private Canvas canvas;
    private GraphicsContext gc;
    private DataManager dataManager;
    private FloodSimulator simulator;
    
    private static final int CANVAS_WIDTH = 1000;
    private static final int CANVAS_HEIGHT = 800;
    
    // Camera parameters
    private double cameraX = 0;
    private double cameraY = 0;
    private double cameraZ = 300;
    private double rotationX = 45;  // degrees
    private double rotationY = 0;   // degrees
    private double zoom = 1.0;
    
    // Mouse interaction
    private double lastMouseX;
    private double lastMouseY;
    
    // Rendering modes
    public enum RenderMode {
        TERRAIN_ONLY,
        FLOOD_HEATMAP,
        ACCESSIBILITY_MAP,
        VELOCITY_VECTORS,
        COMBINED
    }
    
    private RenderMode currentMode = RenderMode.FLOOD_HEATMAP;
    
    // Animation
    private AnimationTimer animationTimer;
    private double simulationTime = 0;
    
    public TerrainRenderer(DataManager dataManager, FloodSimulator simulator) {
        this.dataManager = dataManager;
        this.simulator = simulator;
        
        canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        gc = canvas.getGraphicsContext2D();
        
        setupMouseHandlers();
        setupAnimationTimer();
    }
    
    private void setupMouseHandlers() {
        canvas.setOnMousePressed(this::handleMousePressed);
        canvas.setOnMouseDragged(this::handleMouseDragged);
        canvas.setOnScroll(this::handleScroll);
    }
    
    private void handleMousePressed(MouseEvent e) {
        lastMouseX = e.getX();
        lastMouseY = e.getY();
    }
    
    private void handleMouseDragged(MouseEvent e) {
        double dx = e.getX() - lastMouseX;
        double dy = e.getY() - lastMouseY;
        
        if (e.isSecondaryButtonDown()) {
            // Pan
            cameraX += dx * 0.5;
            cameraY += dy * 0.5;
        } else {
            // Rotate
            rotationY += dx * 0.5;
            rotationX += dy * 0.5;
            rotationX = Math.max(-85, Math.min(85, rotationX));
        }
        
        lastMouseX = e.getX();
        lastMouseY = e.getY();
        
        render();
    }
    
    private void handleScroll(ScrollEvent e) {
        double delta = e.getDeltaY();
        zoom *= (delta > 0) ? 1.1 : 0.9;
        zoom = Math.max(0.1, Math.min(5.0, zoom));
        render();
    }
    
    private void setupAnimationTimer() {
        animationTimer = new AnimationTimer() {
            private long lastUpdate = 0;
            
            @Override
            public void handle(long now) {
                if (lastUpdate == 0) {
                    lastUpdate = now;
                    return;
                }
                
                double deltaSeconds = (now - lastUpdate) / 1_000_000_000.0;
                lastUpdate = now;
                
                // Update simulation time
                if (simulator.isRunning()) {
                    simulationTime += deltaSeconds;
                    simulator.step(deltaSeconds);
                }
                
                render();
            }
        };
    }
    
    public void startAnimation() {
        animationTimer.start();
    }
    
    public void stopAnimation() {
        animationTimer.stop();
    }
    
    /**
     * Main rendering method
     */
    public void render() {
        // Clear canvas
        gc.setFill(Color.rgb(26, 26, 26));
        gc.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        
        // Get terrain data
        TerrainData terrain = dataManager.getTerrainData();
        if (terrain == null) return;
        
        // Render based on current mode
        switch (currentMode) {
            case TERRAIN_ONLY:
                renderTerrain(terrain);
                break;
            case FLOOD_HEATMAP:
                renderTerrainWithFloodHeatmap(terrain);
                break;
            case ACCESSIBILITY_MAP:
                renderAccessibilityMap(terrain);
                break;
            case VELOCITY_VECTORS:
                renderVelocityVectors(terrain);
                break;
            case COMBINED:
                renderCombinedView(terrain);
                break;
        }
        
        // Render buildings
        renderBuildings(terrain);
        
        // Render UI overlays
        renderLegend();
        renderCompass();
        renderTimeIndicator();
    }
    
    /**
     * Render terrain with flood depth heatmap overlay
     */
    private void renderTerrainWithFloodHeatmap(TerrainData terrain) {
        double[][] elevations = terrain.getElevations();
        double[][] floodDepths = simulator.getCurrentFloodDepths();
        
        int rows = elevations.length;
        int cols = elevations[0].length;
        
        double cellSize = 10.0 * zoom; // pixels per cell
        
        // Render from back to front for proper occlusion
        for (int r = rows - 1; r >= 0; r--) {
            for (int c = 0; c < cols; c++) {
                Point3D worldPos = new Point3D(c * 5.0, r * 5.0, elevations[r][c]);
                Point3D screenPos = worldToScreen(worldPos);
                
                // Calculate flood depth at this cell
                double depth = floodDepths[r][c];
                Color cellColor;
                
                if (depth < 0.01) {
                    // No water - show terrain color
                    cellColor = getTerrainColor(elevations[r][c]);
                } else {
                    // Water present - blend with flood color
                    cellColor = getFloodColor(depth);
                }
                
                // Draw cell as parallelogram for 3D effect
                drawIsometricCell(screenPos, cellSize, cellColor, depth);
            }
        }
    }
    
    /**
     * Convert world coordinates to screen coordinates (isometric projection)
     */
    private Point3D worldToScreen(Point3D world) {
        // Apply rotation around Y axis
        double cosY = Math.cos(Math.toRadians(rotationY));
        double sinY = Math.sin(Math.toRadians(rotationY));
        
        double x1 = world.getX() * cosY - world.getY() * sinY;
        double y1 = world.getX() * sinY + world.getY() * cosY;
        double z1 = world.getZ();
        
        // Apply rotation around X axis
        double cosX = Math.cos(Math.toRadians(rotationX));
        double sinX = Math.sin(Math.toRadians(rotationX));
        
        double x2 = x1;
        double y2 = y1 * cosX - z1 * sinX;
        double z2 = y1 * sinX + z1 * cosX;
        
        // Isometric projection with zoom
        double screenX = (CANVAS_WIDTH / 2) + (x2 - y2) * 0.5 * zoom + cameraX;
        double screenY = (CANVAS_HEIGHT / 2) - (x2 + y2) * 0.25 * zoom - z2 * 2.0 * zoom + cameraY;
        
        return new Point3D(screenX, screenY, z2);
    }
    
    /**
     * Draw a single isometric cell
     */
    private void drawIsometricCell(Point3D pos, double size, Color color, double waterDepth) {
        double x = pos.getX();
        double y = pos.getY();
        double w = size * 0.5;
        double h = size * 0.25;
        
        // Draw cell as diamond shape
        double[] xPoints = {x, x + w, x, x - w};
        double[] yPoints = {y - h, y, y + h, y};
        
        gc.setFill(color);
        gc.fillPolygon(xPoints, yPoints, 4);
        
        // Add water shimmer effect for flooded cells
        if (waterDepth > 0.05) {
            double shimmer = Math.sin(simulationTime * 2.0 + x * 0.1 + y * 0.1) * 0.1 + 0.1;
            Color shimmerColor = Color.rgb(255, 255, 255, shimmer);
            gc.setStroke(shimmerColor);
            gc.setLineWidth(1);
            gc.strokePolygon(xPoints, yPoints, 4);
        }
        
        // Draw cell outline
        gc.setStroke(Color.rgb(50, 50, 50, 0.3));
        gc.setLineWidth(0.5);
        gc.strokePolygon(xPoints, yPoints, 4);
    }
    
    /**
     * Get terrain color based on elevation
     */
    private Color getTerrainColor(double elevation) {
        // Normalize elevation (assuming range 0-50m)
        double normalized = Math.max(0, Math.min(1, elevation / 50.0));
        
        // Color gradient: low (dark) to high (light)
        int r = (int)(120 + normalized * 60);
        int g = (int)(100 + normalized * 80);
        int b = (int)(80 + normalized * 60);
        
        return Color.rgb(r, g, b);
    }
    
    /**
     * Get flood color based on water depth (traffic light system)
     */
    private Color getFloodColor(double depth) {
        if (depth < 0.1) {
            // Safe - Green
            return Color.rgb(151, 188, 98, 0.7);
        } else if (depth < 0.3) {
            // Caution - Yellow/Orange
            return Color.rgb(249, 231, 149, 0.8);
        } else if (depth < 0.6) {
            // Warning - Orange
            return Color.rgb(249, 158, 103, 0.85);
        } else {
            // Danger - Red
            return Color.rgb(249, 97, 103, 0.9);
        }
    }
    
    /**
     * Render accessibility map (traffic light zones)
     */
    private void renderAccessibilityMap(TerrainData terrain) {
        double[][] floodDepths = simulator.getCurrentFloodDepths();
        int rows = floodDepths.length;
        int cols = floodDepths[0].length;
        
        // Create accessibility zones
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                Point3D worldPos = new Point3D(c * 5.0, r * 5.0, terrain.getElevations()[r][c]);
                Point3D screenPos = worldToScreen(worldPos);
                
                Color zoneColor = getAccessibilityColor(floodDepths[r][c]);
                double cellSize = 10.0 * zoom;
                
                drawIsometricCell(screenPos, cellSize, zoneColor, floodDepths[r][c]);
            }
        }
    }
    
    private Color getAccessibilityColor(double depth) {
        if (depth < 0.05) {
            return Color.rgb(46, 125, 50, 0.8);  // Green - Accessible
        } else if (depth < 0.15) {
            return Color.rgb(251, 192, 45, 0.85); // Yellow - Caution
        } else {
            return Color.rgb(211, 47, 47, 0.9);   // Red - Inaccessible
        }
    }
    
    /**
     * Render velocity vectors showing water flow direction
     */
    private void renderVelocityVectors(TerrainData terrain) {
        renderTerrain(terrain);
        
        VelocityField velocities = simulator.getCurrentVelocities();
        if (velocities == null) return;
        
        int rows = velocities.getRows();
        int cols = velocities.getCols();
        int skip = 5; // Draw every Nth vector for clarity
        
        gc.setStroke(Color.rgb(255, 255, 255, 0.8));
        gc.setLineWidth(2);
        
        for (int r = 0; r < rows; r += skip) {
            for (int c = 0; c < cols; c += skip) {
                double vx = velocities.getVx(r, c);
                double vy = velocities.getVy(r, c);
                double magnitude = Math.sqrt(vx * vx + vy * vy);
                
                if (magnitude < 0.01) continue; // Skip negligible velocities
                
                Point3D start = new Point3D(c * 5.0, r * 5.0, terrain.getElevations()[r][c]);
                Point3D end = new Point3D((c + vx * 3) * 5.0, (r + vy * 3) * 5.0, 
                                          terrain.getElevations()[r][c]);
                
                Point3D screenStart = worldToScreen(start);
                Point3D screenEnd = worldToScreen(end);
                
                // Draw arrow
                gc.strokeLine(screenStart.getX(), screenStart.getY(), 
                             screenEnd.getX(), screenEnd.getY());
                drawArrowHead(screenStart, screenEnd);
            }
        }
    }
    
    private void drawArrowHead(Point3D start, Point3D end) {
        double dx = end.getX() - start.getX();
        double dy = end.getY() - start.getY();
        double angle = Math.atan2(dy, dx);
        
        double arrowLength = 8;
        double arrowAngle = Math.PI / 6;
        
        double x1 = end.getX() - arrowLength * Math.cos(angle - arrowAngle);
        double y1 = end.getY() - arrowLength * Math.sin(angle - arrowAngle);
        double x2 = end.getX() - arrowLength * Math.cos(angle + arrowAngle);
        double y2 = end.getY() - arrowLength * Math.sin(angle + arrowAngle);
        
        gc.strokeLine(end.getX(), end.getY(), x1, y1);
        gc.strokeLine(end.getX(), end.getY(), x2, y2);
    }
    
    private void renderTerrain(TerrainData terrain) {
        double[][] elevations = terrain.getElevations();
        int rows = elevations.length;
        int cols = elevations[0].length;
        
        for (int r = rows - 1; r >= 0; r--) {
            for (int c = 0; c < cols; c++) {
                Point3D worldPos = new Point3D(c * 5.0, r * 5.0, elevations[r][c]);
                Point3D screenPos = worldToScreen(worldPos);
                Color cellColor = getTerrainColor(elevations[r][c]);
                double cellSize = 10.0 * zoom;
                drawIsometricCell(screenPos, cellSize, cellColor, 0);
            }
        }
    }
    
    private void renderCombinedView(TerrainData terrain) {
        renderTerrainWithFloodHeatmap(terrain);
        renderVelocityVectors(terrain);
    }
    
    /**
     * Render building boundaries
     */
    private void renderBuildings(TerrainData terrain) {
        BuildingData buildings = dataManager.getBuildingData();
        if (buildings == null) return;
        
        gc.setStroke(Color.rgb(33, 41, 92, 0.9));
        gc.setLineWidth(3);
        
        for (Building building : buildings.getBuildings()) {
            double[][] vertices = building.getVertices();
            
            for (int i = 0; i < vertices.length; i++) {
                double x1 = vertices[i][0];
                double y1 = vertices[i][1];
                double z1 = terrain.getElevationAt(x1, y1) + 5.0; // Buildings are 5m above ground
                
                int nextIdx = (i + 1) % vertices.length;
                double x2 = vertices[nextIdx][0];
                double y2 = vertices[nextIdx][1];
                double z2 = terrain.getElevationAt(x2, y2) + 5.0;
                
                Point3D screen1 = worldToScreen(new Point3D(x1, y1, z1));
                Point3D screen2 = worldToScreen(new Point3D(x2, y2, z2));
                
                gc.strokeLine(screen1.getX(), screen1.getY(), 
                             screen2.getX(), screen2.getY());
            }
        }
    }
    
    /**
     * Render legend for flood depth / accessibility
     */
    private void renderLegend() {
        double legendX = CANVAS_WIDTH - 180;
        double legendY = 20;
        double legendWidth = 160;
        double legendHeight = 200;
        
        // Background
        gc.setFill(Color.rgb(30, 30, 30, 0.9));
        gc.fillRoundRect(legendX, legendY, legendWidth, legendHeight, 10, 10);
        
        // Title
        gc.setFill(Color.WHITE);
        gc.setFont(javafx.scene.text.Font.font("Arial", 14));
        gc.fillText("Flood Depth (m)", legendX + 10, legendY + 25);
        
        // Color bars
        double barStartY = legendY + 40;
        double barHeight = 30;
        
        String[] labels = {"0.0-0.1 (Safe)", "0.1-0.3 (Caution)", 
                          "0.3-0.6 (Warning)", ">0.6 (Danger)"};
        Color[] colors = {
            Color.rgb(151, 188, 98),
            Color.rgb(249, 231, 149),
            Color.rgb(249, 158, 103),
            Color.rgb(249, 97, 103)
        };
        
        gc.setFont(javafx.scene.text.Font.font("Arial", 11));
        
        for (int i = 0; i < labels.length; i++) {
            double y = barStartY + i * (barHeight + 5);
            
            // Color bar
            gc.setFill(colors[i]);
            gc.fillRect(legendX + 10, y, 30, barHeight);
            
            // Label
            gc.setFill(Color.WHITE);
            gc.fillText(labels[i], legendX + 50, y + 20);
        }
    }
    
    /**
     * Render compass for orientation
     */
    private void renderCompass() {
        double compassX = 50;
        double compassY = 50;
        double compassSize = 40;
        
        // Background circle
        gc.setFill(Color.rgb(30, 30, 30, 0.8));
        gc.fillOval(compassX - compassSize, compassY - compassSize, 
                    compassSize * 2, compassSize * 2);
        
        // North arrow
        double arrowAngle = Math.toRadians(-rotationY);
        double arrowX = compassX + compassSize * 0.7 * Math.sin(arrowAngle);
        double arrowY = compassY - compassSize * 0.7 * Math.cos(arrowAngle);
        
        gc.setStroke(Color.rgb(249, 97, 103));
        gc.setLineWidth(3);
        gc.strokeLine(compassX, compassY, arrowX, arrowY);
        
        // N label
        gc.setFill(Color.WHITE);
        gc.setFont(javafx.scene.text.Font.font("Arial", 12));
        gc.fillText("N", arrowX - 5, arrowY - 5);
    }
    
    /**
     * Render simulation time indicator
     */
    private void renderTimeIndicator() {
        if (!simulator.isRunning()) return;
        
        double x = CANVAS_WIDTH / 2 - 100;
        double y = 30;
        
        gc.setFill(Color.rgb(6, 90, 130, 0.9));
        gc.fillRoundRect(x, y, 200, 40, 10, 10);
        
        gc.setFill(Color.WHITE);
        gc.setFont(javafx.scene.text.Font.font("Arial", 16));
        
        int hours = (int) (simulationTime / 3600);
        int minutes = (int) ((simulationTime % 3600) / 60);
        int seconds = (int) (simulationTime % 60);
        
        String timeStr = String.format("Time: %02d:%02d:%02d", hours, minutes, seconds);
        gc.fillText(timeStr, x + 20, y + 25);
    }
    
    // Getters and setters
    public Canvas getCanvas() { return canvas; }
    public void setRenderMode(RenderMode mode) { 
        this.currentMode = mode; 
        render();
    }
    public RenderMode getRenderMode() { return currentMode; }
    public void resetCamera() {
        cameraX = 0;
        cameraY = 0;
        rotationX = 45;
        rotationY = 0;
        zoom = 1.0;
        render();
    }
}
