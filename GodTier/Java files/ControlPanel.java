import javafx.scene.layout.VBox;
import javafx.scene.control.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;

/**
 * Interactive Control Panel for Flood Simulation
 * Provides user controls for:
 * - Rainfall scenarios
 * - Simulation parameters
 * - Visualization modes
 * - Camera controls
 */
public class ControlPanel {
    
    private VBox panel;
    private TerrainRenderer renderer;
    private FloodSimulator simulator;
    private DataManager dataManager;
    
    // UI Components
    private ComboBox<String> villageSelector;
    private ComboBox<String> rainfallSelector;
    private Slider rainfallSlider;
    private Label rainfallLabel;
    private Button startButton;
    private Button stopButton;
    private Button resetButton;
    private ComboBox<String> visualizationMode;
    private Slider infiltrationSlider;
    private Slider manningSlider;
    private Label infiltrationLabel;
    private Label manningLabel;
    private CheckBox showBuildingsCheck;
    private CheckBox showVectorsCheck;
    
    private static final int PANEL_WIDTH = 350;
    
    public ControlPanel(TerrainRenderer renderer, FloodSimulator simulator, DataManager dataManager) {
        this.renderer = renderer;
        this.simulator = simulator;
        this.dataManager = dataManager;
        
        createPanel();
    }
    
    private void createPanel() {
        panel = new VBox(15);
        panel.setPadding(new Insets(20));
        panel.setPrefWidth(PANEL_WIDTH);
        panel.setStyle("-fx-background-color: #2a2a2a; -fx-border-color: #065A82; -fx-border-width: 0 0 0 2;");
        
        // Title
        Label title = new Label("Simulation Controls");
        title.setStyle("-fx-text-fill: white; -fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Village Selection Section
        VBox villageSection = createVillageSection();
        
        // Rainfall Configuration Section
        VBox rainfallSection = createRainfallSection();
        
        // Simulation Control Section
        VBox controlSection = createControlSection();
        
        // Visualization Section
        VBox visualizationSection = createVisualizationSection();
        
        // Advanced Parameters Section
        VBox advancedSection = createAdvancedSection();
        
        // Camera Controls Section
        VBox cameraSection = createCameraSection();
        
        // Add separator
        Separator sep1 = new Separator();
        Separator sep2 = new Separator();
        Separator sep3 = new Separator();
        Separator sep4 = new Separator();
        
        // Add all sections to panel
        panel.getChildren().addAll(
            title,
            new Separator(),
            villageSection,
            sep1,
            rainfallSection,
            sep2,
            controlSection,
            sep3,
            visualizationSection,
            sep4,
            advancedSection,
            new Separator(),
            cameraSection
        );
        
        // Make it scrollable
        ScrollPane scrollPane = new ScrollPane(panel);
        scrollPane.setFitToWidth(true);
        scrollPane.setStyle("-fx-background-color: #2a2a2a; -fx-border-color: transparent;");
        
        // This is the actual panel that gets added to the main layout
        VBox wrapper = new VBox(scrollPane);
        VBox.setVgrow(scrollPane, Priority.ALWAYS);
        this.panel = wrapper;
    }
    
    private VBox createVillageSection() {
        VBox section = new VBox(8);
        
        Label sectionLabel = createSectionLabel("Village Selection");
        
        villageSelector = new ComboBox<>();
        villageSelector.getItems().addAll(
            "Dariyapur (UP) - 118118",
            "Devipura (UP) - 118125",
            "Manjhola Khurd (UP) - 118129"
        );
        villageSelector.setValue("Dariyapur (UP) - 118118");
        villageSelector.setPrefWidth(300);
        villageSelector.setStyle("-fx-background-color: #3a3a3a; -fx-text-fill: white;");
        
        villageSelector.setOnAction(e -> loadSelectedVillage());
        
        section.getChildren().addAll(sectionLabel, villageSelector);
        return section;
    }
    
    private VBox createRainfallSection() {
        VBox section = new VBox(8);
        
        Label sectionLabel = createSectionLabel("Rainfall Scenario");
        
        // Predefined scenarios
        rainfallSelector = new ComboBox<>();
        rainfallSelector.getItems().addAll(
            "Light Rain (10 mm/hr)",
            "Moderate Rain (30 mm/hr)",
            "Heavy Rain (75 mm/hr)",
            "Extreme Event (150 mm/hr)",
            "Custom Intensity"
        );
        rainfallSelector.setValue("Moderate Rain (30 mm/hr)");
        rainfallSelector.setPrefWidth(300);
        rainfallSelector.setStyle("-fx-background-color: #3a3a3a; -fx-text-fill: white;");
        
        rainfallSelector.setOnAction(e -> updateRainfallFromSelector());
        
        // Custom rainfall slider
        HBox sliderBox = new HBox(10);
        rainfallLabel = new Label("Intensity: 30 mm/hr");
        rainfallLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 11px;");
        
        rainfallSlider = new Slider(5, 200, 30);
        rainfallSlider.setShowTickLabels(false);
        rainfallSlider.setShowTickMarks(true);
        rainfallSlider.setMajorTickUnit(50);
        rainfallSlider.setBlockIncrement(5);
        rainfallSlider.setPrefWidth(200);
        
        rainfallSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            rainfallLabel.setText(String.format("Intensity: %.0f mm/hr", newVal.doubleValue()));
            simulator.setRainfallRate(newVal.doubleValue());
            rainfallSelector.setValue("Custom Intensity");
        });
        
        sliderBox.getChildren().addAll(rainfallLabel, rainfallSlider);
        sliderBox.setAlignment(Pos.CENTER_LEFT);
        
        section.getChildren().addAll(sectionLabel, rainfallSelector, sliderBox);
        return section;
    }
    
    private VBox createControlSection() {
        VBox section = new VBox(10);
        
        Label sectionLabel = createSectionLabel("Simulation Control");
        
        HBox buttonBox = new HBox(10);
        
        startButton = new Button("▶ Start");
        startButton.setStyle("-fx-background-color: #2E7D32; -fx-text-fill: white; " +
                            "-fx-font-weight: bold; -fx-cursor: hand;");
        startButton.setPrefWidth(100);
        startButton.setOnAction(e -> {
            simulator.start();
            updateButtonStates();
        });
        
        stopButton = new Button("⏸ Pause");
        stopButton.setStyle("-fx-background-color: #F57C00; -fx-text-fill: white; " +
                           "-fx-font-weight: bold; -fx-cursor: hand;");
        stopButton.setPrefWidth(100);
        stopButton.setDisable(true);
        stopButton.setOnAction(e -> {
            simulator.stop();
            updateButtonStates();
        });
        
        resetButton = new Button("⟲ Reset");
        resetButton.setStyle("-fx-background-color: #C62828; -fx-text-fill: white; " +
                            "-fx-font-weight: bold; -fx-cursor: hand;");
        resetButton.setPrefWidth(100);
        resetButton.setOnAction(e -> {
            simulator.reset();
            simulator.stop();
            updateButtonStates();
        });
        
        buttonBox.getChildren().addAll(startButton, stopButton, resetButton);
        
        // Statistics display
        VBox statsBox = new VBox(5);
        statsBox.setPadding(new Insets(10));
        statsBox.setStyle("-fx-background-color: #1a1a1a; -fx-border-color: #065A82; " +
                         "-fx-border-width: 1; -fx-border-radius: 5;");
        
        Label statsTitle = new Label("Current Statistics");
        statsTitle.setStyle("-fx-text-fill: #1C7293; -fx-font-size: 12px; -fx-font-weight: bold;");
        
        Label maxDepthLabel = new Label("Max Depth: 0.00 m");
        maxDepthLabel.setStyle("-fx-text-fill: white; -fx-font-size: 11px;");
        
        Label avgDepthLabel = new Label("Avg Depth: 0.00 m");
        avgDepthLabel.setStyle("-fx-text-fill: white; -fx-font-size: 11px;");
        
        Label floodedAreaLabel = new Label("Flooded Area: 0%");
        floodedAreaLabel.setStyle("-fx-text-fill: white; -fx-font-size: 11px;");
        
        statsBox.getChildren().addAll(statsTitle, maxDepthLabel, avgDepthLabel, floodedAreaLabel);
        
        section.getChildren().addAll(sectionLabel, buttonBox, statsBox);
        return section;
    }
    
    private VBox createVisualizationSection() {
        VBox section = new VBox(8);
        
        Label sectionLabel = createSectionLabel("Visualization Mode");
        
        visualizationMode = new ComboBox<>();
        visualizationMode.getItems().addAll(
            "Flood Heatmap",
            "Accessibility Map",
            "Velocity Vectors",
            "Terrain Only",
            "Combined View"
        );
        visualizationMode.setValue("Flood Heatmap");
        visualizationMode.setPrefWidth(300);
        visualizationMode.setStyle("-fx-background-color: #3a3a3a; -fx-text-fill: white;");
        
        visualizationMode.setOnAction(e -> updateVisualizationMode());
        
        // Display options
        showBuildingsCheck = new CheckBox("Show Buildings");
        showBuildingsCheck.setSelected(true);
        showBuildingsCheck.setStyle("-fx-text-fill: white;");
        
        showVectorsCheck = new CheckBox("Show Flow Vectors");
        showVectorsCheck.setSelected(false);
        showVectorsCheck.setStyle("-fx-text-fill: white;");
        
        section.getChildren().addAll(sectionLabel, visualizationMode, 
                                     showBuildingsCheck, showVectorsCheck);
        return section;
    }
    
    private VBox createAdvancedSection() {
        VBox section = new VBox(8);
        
        Label sectionLabel = createSectionLabel("Advanced Parameters");
        
        // Infiltration rate
        HBox infiltrationBox = new HBox(10);
        infiltrationLabel = new Label("Infiltration: 5 mm/hr");
        infiltrationLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 11px;");
        
        infiltrationSlider = new Slider(0, 20, 5);
        infiltrationSlider.setShowTickMarks(true);
        infiltrationSlider.setMajorTickUnit(5);
        infiltrationSlider.setPrefWidth(180);
        
        infiltrationSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            infiltrationLabel.setText(String.format("Infiltration: %.1f mm/hr", newVal.doubleValue()));
            simulator.setInfiltrationRate(newVal.doubleValue());
        });
        
        infiltrationBox.getChildren().addAll(infiltrationLabel, infiltrationSlider);
        infiltrationBox.setAlignment(Pos.CENTER_LEFT);
        
        // Manning coefficient
        HBox manningBox = new HBox(10);
        manningLabel = new Label("Roughness: 0.030");
        manningLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 11px;");
        
        manningSlider = new Slider(0.015, 0.050, 0.030);
        manningSlider.setShowTickMarks(true);
        manningSlider.setMajorTickUnit(0.010);
        manningSlider.setPrefWidth(180);
        
        manningSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            manningLabel.setText(String.format("Roughness: %.3f", newVal.doubleValue()));
            simulator.setManningCoefficient(newVal.doubleValue());
        });
        
        manningBox.getChildren().addAll(manningLabel, manningSlider);
        manningBox.setAlignment(Pos.CENTER_LEFT);
        
        Label helpText = new Label("Roughness: 0.015=smooth, 0.030=normal, 0.050=rough");
        helpText.setStyle("-fx-text-fill: #888888; -fx-font-size: 9px; -fx-font-style: italic;");
        helpText.setWrapText(true);
        
        section.getChildren().addAll(sectionLabel, infiltrationBox, manningBox, helpText);
        return section;
    }
    
    private VBox createCameraSection() {
        VBox section = new VBox(8);
        
        Label sectionLabel = createSectionLabel("Camera Controls");
        
        Button resetCameraButton = new Button("Reset View");
        resetCameraButton.setStyle("-fx-background-color: #1C7293; -fx-text-fill: white; " +
                                  "-fx-font-weight: bold; -fx-cursor: hand;");
        resetCameraButton.setPrefWidth(150);
        resetCameraButton.setOnAction(e -> renderer.resetCamera());
        
        Label helpLabel = new Label(
            "• Left-drag: Rotate\n" +
            "• Right-drag: Pan\n" +
            "• Scroll: Zoom"
        );
        helpLabel.setStyle("-fx-text-fill: #CADCFC; -fx-font-size: 10px;");
        
        section.getChildren().addAll(sectionLabel, resetCameraButton, helpLabel);
        return section;
    }
    
    private Label createSectionLabel(String text) {
        Label label = new Label(text);
        label.setStyle("-fx-text-fill: #1C7293; -fx-font-size: 13px; -fx-font-weight: bold;");
        return label;
    }
    
    private void updateRainfallFromSelector() {
        String selected = rainfallSelector.getValue();
        
        if (selected.contains("Light")) {
            rainfallSlider.setValue(10);
            simulator.setRainfallScenario(FloodSimulator.RainfallScenario.LIGHT);
        } else if (selected.contains("Moderate")) {
            rainfallSlider.setValue(30);
            simulator.setRainfallScenario(FloodSimulator.RainfallScenario.MODERATE);
        } else if (selected.contains("Heavy")) {
            rainfallSlider.setValue(75);
            simulator.setRainfallScenario(FloodSimulator.RainfallScenario.HEAVY);
        } else if (selected.contains("Extreme")) {
            rainfallSlider.setValue(150);
            simulator.setRainfallScenario(FloodSimulator.RainfallScenario.EXTREME);
        }
    }
    
    private void updateVisualizationMode() {
        String mode = visualizationMode.getValue();
        
        if (mode.contains("Heatmap")) {
            renderer.setRenderMode(TerrainRenderer.RenderMode.FLOOD_HEATMAP);
        } else if (mode.contains("Accessibility")) {
            renderer.setRenderMode(TerrainRenderer.RenderMode.ACCESSIBILITY_MAP);
        } else if (mode.contains("Velocity")) {
            renderer.setRenderMode(TerrainRenderer.RenderMode.VELOCITY_VECTORS);
        } else if (mode.contains("Terrain")) {
            renderer.setRenderMode(TerrainRenderer.RenderMode.TERRAIN_ONLY);
        } else if (mode.contains("Combined")) {
            renderer.setRenderMode(TerrainRenderer.RenderMode.COMBINED);
        }
    }
    
    private void loadSelectedVillage() {
        String selected = villageSelector.getValue();
        
        if (selected.contains("Dariyapur")) {
            dataManager.loadVillageData("data/dariyapur_dtm.csv", 
                                       "data/dariyapur_buildings.geojson",
                                       "data/dariyapur_simulation.bin");
        } else if (selected.contains("Devipura")) {
            dataManager.loadVillageData("data/devipura_dtm.csv", 
                                       "data/devipura_buildings.geojson",
                                       "data/devipura_simulation.bin");
        } else if (selected.contains("Manjhola")) {
            dataManager.loadVillageData("data/manjhola_dtm.csv", 
                                       "data/manjhola_buildings.geojson",
                                       "data/manjhola_simulation.bin");
        }
        
        simulator.reset();
        renderer.resetCamera();
    }
    
    private void updateButtonStates() {
        boolean running = simulator.isRunning();
        startButton.setDisable(running);
        stopButton.setDisable(!running);
    }
    
    public VBox getPanel() {
        return panel;
    }
}
