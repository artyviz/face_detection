# InsightFace Real-Time Attendance System

A lightweight, real-time face-recognition attendance system that works with **any USB webcam** or **IP camera**.
Built with **InsightFace** + **OpenCV tracking** + **CSV logging**.

---

## 🚀 Features

* Threaded Frame Capture
* Bounded Queue for Memory Safety
* Object Tracking to Skip Redundant Detection
* GPU-Ready InsightFace Backend
* CSV Logging with Duplicate Prevention

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## 🛠️ Quick Start

### Add faces

* Put portrait images in `./images/`
* Filename → person’s name (e.g., `Alice.jpg`)

### Run

```bash
python src/main.py
```

* Press `q` to quit.
* Attendance is saved to `attendance.csv`.

---

## ⚙️ Configuration

Edit constants at the top of `main.py`:

| Key            | Default  | Meaning               |
| -------------- | -------- | --------------------- |
| IMAGE\_DIR     | ./images | Gallery folder        |
| SIM\_THRESHOLD | 0.35     | Face-match confidence |
| USE\_GPU\_CTX  | 0        | GPU id (-1 for CPU)   |

---

## 🤝 Contributing

PRs welcome! Please open an issue first.

---

## 📄 License

MIT © Farhan
