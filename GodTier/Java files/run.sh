#!/bin/bash

# ================================================================
# Eco-Hydro-AI Platform - Run Script
# Team Dupahar | Innovathon 1.0
# Launches the application with JavaFX + JSON library
# ================================================================

echo "================================================"
echo "   Eco-Hydro-AI Platform - Launcher"
echo "================================================"
echo ""

# ================= Configuration =================
JAVAFX_PATH="/usr/local/javafx-sdk-19/lib"
JAVAFX_MODULES="javafx.controls,javafx.graphics"
JSON_JAR="json-20231013.jar"
MAIN_CLASS="FloodVisualizationApp"
BUILD_DIR="build"

# ================= Checks =================
if [ ! -d "$JAVAFX_PATH" ]; then
    echo "❌ JavaFX SDK not found at: $JAVAFX_PATH"
    echo "Update JAVAFX_PATH inside run.sh"
    echo "Download from: https://openjfx.io/"
    exit 1
fi

if [ ! -f "$JSON_JAR" ]; then
    echo "❌ JSON library not found: $JSON_JAR"
    echo "Place the jar inside project folder."
    exit 1
fi

# ================= Auto Build =================
if [ ! -f "$BUILD_DIR/$MAIN_CLASS.class" ]; then
    echo "⚠️  Compiled files not found."
    echo "🔧 Running build script..."
    echo ""

    ./build.sh

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Build failed. Cannot run application."
        exit 1
    fi
fi

# ================= Launch Info =================
echo "🚀 Launching Eco-Hydro-AI Platform..."
echo "   JavaFX Path : $JAVAFX_PATH"
echo "   JSON Library: $JSON_JAR"
echo ""
echo "📝 Controls:"
echo "   • Left-drag : Rotate camera"
echo "   • Right-drag: Pan view"
echo "   • Scroll    : Zoom in/out"
echo ""
echo "================================================"
echo ""

# ================= Run Application =================
java \
  --module-path "$JAVAFX_PATH" \
  --add-modules "$JAVAFX_MODULES" \
  -cp "$BUILD_DIR:$JSON_JAR" \
  "$MAIN_CLASS"

# ================= Result =================
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application encountered an error!"
    echo "Check messages above."
    echo ""
    exit 1
fi
