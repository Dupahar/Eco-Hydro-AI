#!/bin/bash

# ================================================================
# Eco-Hydro-AI Platform - Production Build Script
# Team Dupahar | Innovathon 1.0
# Compiles all Java source files with JavaFX + JSON library
# ================================================================

echo "================================================"
echo "   Eco-Hydro-AI Platform - Build Script"
echo "================================================"
echo ""

# ================= Configuration =================
JAVAFX_PATH="/usr/local/javafx-sdk-19/lib"
JAVAFX_MODULES="javafx.controls,javafx.graphics"
JSON_JAR="json-20231013.jar"

# Compile all java files automatically
SOURCE_FILES="*.java"

# ================= Checks =================
if [ ! -d "$JAVAFX_PATH" ]; then
    echo "❌ JavaFX SDK not found at: $JAVAFX_PATH"
    echo "Update JAVAFX_PATH inside build.sh"
    echo "Download JavaFX from: https://openjfx.io/"
    exit 1
fi

if [ ! -f "$JSON_JAR" ]; then
    echo "❌ JSON library not found: $JSON_JAR"
    echo "Place the jar inside project folder."
    exit 1
fi

# Create build folder
mkdir -p build

echo "🔧 Compiling Java source files..."
echo "   JavaFX Path : $JAVAFX_PATH"
echo "   JSON Library: $JSON_JAR"
echo ""

# ================= Compile =================
javac \
  --module-path "$JAVAFX_PATH" \
  --add-modules "$JAVAFX_MODULES" \
  -cp ".:$JSON_JAR" \
  -d build \
  $SOURCE_FILES

# ================= Result =================
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
    echo "▶ Run with:"
    echo "java --module-path $JAVAFX_PATH \\"
    echo "     --add-modules $JAVAFX_MODULES \\"
    echo "     -cp \"build:$JSON_JAR\" \\"
    echo "     FloodVisualizationApp"
    echo ""
else
    echo ""
    echo "❌ Compilation failed!"
    echo "Check error messages above."
    echo ""
    exit 1
fi

echo "================================================"
