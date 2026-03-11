import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import javafx.scene.control.*;
import javafx.geometry.Insets;
import javafx.scene.layout.VBox;
import javafx.scene.layout.HBox;

/**
 * Eco-Hydro-AI Platform - Advanced Flood Visualization System
 * Main Application Entry Point
 * 
 * Features:
 * - 3D Terrain Visualization
 * - Real-time Flood Simulation
 * - Interactive Heatmap Display
 * - Traffic Light Accessibility Analysis
 * - Physics-informed Water Flow Animation
 * 
 * @author Team Dupahar
 * @version 1.0 - Innovathon 2026
 */
public class FloodVisualizationApp extends Application {
    
    private TerrainRenderer terrainRenderer;
    private FloodSimulator floodSimulator;
    private ControlPanel controlPanel;
    private DataManager dataManager;
    
    private static final String APP_TITLE = "Eco-Hydro-AI Platform - Flood Visualization";
    private static final int WINDOW_WIDTH = 1400;
    private static final int WINDOW_HEIGHT = 900;
    
    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle(APP_TITLE);
        
        // Initialize core components
        dataManager = new DataManager();
        floodSimulator = new FloodSimulator(dataManager);
        terrainRenderer = new TerrainRenderer(dataManager, floodSimulator);
        controlPanel = new ControlPanel(terrainRenderer, floodSimulator, dataManager);
        
        // Create main layout
        BorderPane root = new BorderPane();
        root.setCenter(terrainRenderer.getCanvas());
        root.setRight(controlPanel.getPanel());
        root.setBottom(createStatusBar());
        
        // Apply styling
        root.setStyle("-fx-background-color: #1a1a1a;");
        
        // Create scene and show
        Scene scene = new Scene(root, WINDOW_WIDTH, WINDOW_HEIGHT);
        scene.getStylesheets().add(getClass().getResource("styles.css").toExternalForm());
        
        primaryStage.setScene(scene);
        primaryStage.show();
        
        // Load default village data
        loadDefaultVillage();
        
        // Start animation loop
        terrainRenderer.startAnimation();
    }
    
    private HBox createStatusBar() {
        HBox statusBar = new HBox(20);
        statusBar.setPadding(new Insets(10));
        statusBar.setStyle("-fx-background-color: #065A82; -fx-text-fill: white;");
        
        Label villageLabel = new Label("Village: Dariyapur (Survey ID: 118118)");
        villageLabel.setStyle("-fx-text-fill: white; -fx-font-size: 12px; -fx-font-weight: bold;");
        
        Label resolutionLabel = new Label("Resolution: 3.5cm GSD");
        resolutionLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 12px;");
        
        Label nodesLabel = new Label("Graph Nodes: 7.5M");
        nodesLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 12px;");
        
        Label accuracyLabel = new Label("Model MAE: 3.3cm");
        accuracyLabel.setStyle("-fx-text-fill: #97BC62; -fx-font-size: 12px; -fx-font-weight: bold;");
        
        statusBar.getChildren().addAll(villageLabel, new Separator(), 
                                       resolutionLabel, nodesLabel, accuracyLabel);
        
        return statusBar;
    }
    
    private void loadDefaultVillage() {
        // Load pre-computed village data
        dataManager.loadVillageData("data/dariyapur_dtm.csv", 
                                     "data/dariyapur_buildings.geojson",
                                     "data/dariyapur_simulation.bin");
    }
    
    public static void main(String[] args) {
        launch(args);
    }
}
