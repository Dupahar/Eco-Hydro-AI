#!/bin/bash

# Eco-Hydro-AI Platform Build Script
# Compiles all Java source files with JavaFX

echo "================================================"
echo "   Eco-Hydro-AI Platform - Build Script"
echo "   Team Dupahar | Innovathon 1.0"
echo "================================================"
echo ""

# Configuration
JAVAFX_PATH="/usr/local/javafx-sdk-19/lib"
JAVAFX_MODULES="javafx.controls,javafx.graphics"
SOURCE_FILES="FloodVisualizationApp.java TerrainRenderer.java FloodSimulator.java ControlPanel.java DataStructures.java"

# Check if JavaFX path exists
if [ ! -d "$JAVAFX_PATH" ]; then
    echo "❌ JavaFX SDK not found at: $JAVAFX_PATH"
    echo ""
    echo "Please update JAVAFX_PATH in build.sh to point to your JavaFX installation"
    echo "Download from: https://openjfx.io/"
    echo ""
    exit 1
fi

echo "🔧 Compiling Java source files..."
echo "   JavaFX Path: $JAVAFX_PATH"
echo ""

# Compile
javac --module-path "$JAVAFX_PATH" \
      --add-modules "$JAVAFX_MODULES" \
      $SOURCE_FILES

# Check compilation result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
    echo "Run the application with:"
    echo "   ./run.sh"
    echo ""
    echo "Or manually:"
    echo "   java --module-path $JAVAFX_PATH \\"
    echo "        --add-modules $JAVAFX_MODULES \\"
    echo "        FloodVisualizationApp"
    echo ""
else
    echo ""
    echo "❌ Compilation failed!"
    echo "   Please check error messages above"
    echo ""
    exit 1
fi

echo "================================================"
