# 🎨 Virtual Air Canvas

**100% real. 0% mouse. 1 finger. Infinite artistic confidence.**

Draw anything (even Mona Lisa!) just by moving your finger in front of the camera. Built with MediaPipe + OpenCV using a regular webcam. Zero hardware required. Pure magic. ✨

![Air Canvas Demo](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)

---

## ✨ Features

- ✏️ **Draw freely in the air** using index finger tracking
- 🤏 **Pinch gesture** (thumb + index) to click virtual buttons
- 🎨 **7 Colors**: Red, Green, Blue, Yellow, Magenta, Cyan, White
- 🗑️ **Clear canvas** button
- ⏸️ **Toggle pen ON/OFF** with `T` key
- 📏 **Adjustable brush thickness** with `+` and `-` keys
- 👁️ **Visual feedback**: yellow hover, green flash on button press
- 🖼️ **Clean virtual canvas** window separate from camera feed

---

## 🐍 Recommended Python Version

**Python 3.9, 3.10, or 3.11** (tested and working perfectly)

> ⚠️ **Avoid Python 3.12+** - MediaPipe has compatibility issues with newer Python versions as of now.

**Best choice: Python 3.10.11** ✅

---

## 📦 Step-by-Step Installation Guide (VS Code)

### **Step 1: Install Python**

1. Download Python 3.10.11 from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"** ✅
3. Verify installation:
   ```bash
   python --version
   ```
   Should show: `Python 3.10.11`

---

### **Step 2: Install VS Code**

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install the **Python extension** by Microsoft in VS Code

---

### **Step 3: Create Project Folder**

1. Create a new folder (e.g., `virtual-air-canvas`)
2. Open it in VS Code: `File → Open Folder`

---

### **Step 4: Create Project Files**

Create these 3 files in your project folder:

#### **File 1: `main.py`**
Copy the complete Python code from the first artifact above.

#### **File 2: `requirements.txt`**
Copy the dependencies from the second artifact above.

#### **File 3: `README.md`** (optional)
Copy this guide for future reference.

---

### **Step 5: Create Virtual Environment**

Open VS Code terminal (`Ctrl + ~` or `` Ctrl + ` ``) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### **Step 6: Install Dependencies**

With virtual environment activated:

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python (computer vision library)
- mediapipe (hand tracking)
- numpy (numerical operations)

Wait for installation to complete (~2-3 minutes).

---

### **Step 7: Run the Application**

```bash
python main.py
```

**First time?** Your system may ask for **camera permission** - click **Allow**.

---

## 🎮 Controls

### Keyboard Controls

| Key | Action |
|-----|--------|
| `T` / `t` | Toggle drawing mode (Pen ON/OFF) |
| `+` / `=` | Increase brush thickness |
| `-` / `_` | Decrease brush thickness |
| `Q` / `q` | Quit the application |

### Hand Gestures

| Gesture | Action |
|---------|--------|
| **Pinch** (thumb + index close) | Click virtual buttons |
| **Index finger** (pen active) | Draw on canvas |
| **Move finger anywhere** | Control cursor |

---

## 🖼️ How It Works

1. **MediaPipe Hands** detects 21 hand landmarks in real-time
2. **Pinch Detection**: Distance between thumb tip and index fingertip < 40 pixels
3. **Drawing Cursor**: Index fingertip position (landmark #8)
4. **Virtual Buttons**: Placed on right side of camera feed (1150px x-offset)
5. **Persistent Canvas**: Separate drawing layer that persists across frames

---

## 🚨 Troubleshooting

### **Problem: Camera not detected**
```bash
# Test camera separately
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### **Problem: ModuleNotFoundError**
```bash
# Ensure virtual environment is active (you see (venv) in terminal)
# Then reinstall:
pip install -r requirements.txt
```

### **Problem: Slow/laggy performance**
- Close other applications using webcam
- Lower `min_detection_confidence` to `0.5` in code (line 136)
- Reduce webcam resolution in code (line 124-125)

### **Problem: Hand not detected**
- Ensure good lighting
- Keep hand 1-2 feet from camera
- Show full hand to camera (all fingers visible)

---

## 💡 Ideas for Future Enhancements

**Contributions Welcome!** 🎉

- [ ] **Eraser mode** - Toggle between pen and eraser
- [ ] **Multiple brush styles** - Marker, spray, watercolor effects
- [ ] **Save drawings** - Export canvas to PNG/JPG file
- [ ] **Advanced color picker** - Use palm or multiple fingers for color wheel
- [ ] **Undo/Redo** - Step backward/forward in drawing history
- [ ] **Shape tools** - Draw perfect circles, rectangles, lines
- [ ] **Background images** - Import reference images to trace
- [ ] **Recording mode** - Save timelapse video of drawing process
- [ ] **Multi-hand support** - Use both hands simultaneously
- [ ] **Gesture shortcuts** - Peace sign = screenshot, fist = undo

---

## 📁 Project Structure

```
virtual-air-canvas/
│
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── venv/               # Virtual environment (created by you)
```

---

## 🙏 Credits & Technologies

- **MediaPipe** - Google's ML solution for hand tracking
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing library

---

## 📄 License

This project is open source and available for educational purposes. Feel free to modify and enhance! ⭐

---

## 🎬 Demo Tips

Want to draw the Mona Lisa? Start with basic shapes:
1. Toggle pen ON (`T` key)
2. Select color (pinch on color button)
3. Draw outline with thin brush (`-` key)
4. Switch colors and fill details
5. Increase thickness for bold strokes (`+` key)

**Pro tip**: Look at the "Drawing" window (not camera feed) to see clean output!

---

## 🤝 Contributing

Found a bug? Have an enhancement idea? 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Built with ❤️ using Python, OpenCV, and MediaPipe**

*Now go draw something amazing! 🚀*