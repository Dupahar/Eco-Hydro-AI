#!/bin/bash

# Eco-Hydro-AI Platform Run Script
# Executes the compiled application with JavaFX

echo "================================================"
echo "   Eco-Hydro-AI Platform - Launcher"
echo "   Team Dupahar | Innovathon 1.0"
echo "================================================"
echo ""

# Configuration
JAVAFX_PATH="/usr/local/javafx-sdk-19/lib"
JAVAFX_MODULES="javafx.controls,javafx.graphics"
MAIN_CLASS="FloodVisualizationApp"

# Check if JavaFX path exists
if [ ! -d "$JAVAFX_PATH" ]; then
    echo "❌ JavaFX SDK not found at: $JAVAFX_PATH"
    echo ""
    echo "Please update JAVAFX_PATH in run.sh to point to your JavaFX installation"
    echo "Download from: https://openjfx.io/"
    echo ""
    exit 1
fi

# Check if compiled
if [ ! -f "$MAIN_CLASS.class" ]; then
    echo "⚠️  Application not compiled yet!"
    echo ""
    echo "Running build script..."
    echo ""
    ./build.sh
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Build failed. Cannot run application."
        exit 1
    fi
fi

echo "🚀 Launching Eco-Hydro-AI Platform..."
echo "   JavaFX Path: $JAVAFX_PATH"
echo ""
echo "📝 Controls:"
echo "   • Left-drag: Rotate camera"
echo "   • Right-drag: Pan view"
echo "   • Scroll: Zoom in/out"
echo ""
echo "================================================"
echo ""

# Run application
java --module-path "$JAVAFX_PATH" \
     --add-modules "$JAVAFX_MODULES" \
     "$MAIN_CLASS"

# Check execution result
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application encountered an error!"
    echo "   Check error messages above"
    echo ""
    exit 1
fi
