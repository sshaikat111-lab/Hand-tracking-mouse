STEP 1: Install Python

Download and install Python from:
https://www.python.org/downloads/

During installation, check:
☑ Add Python to PATH

---

STEP 2: Open Command Prompt

Press:
Win + R

Type:
cmd

Press Enter

---

STEP 3: Install Required Libraries

Copy and paste:

pip install opencv-python cvzone pyautogui mediapipe

Wait until installation is complete.

---

STEP 4: Create the Python File

Open Notepad.

Paste the Python code.

Save the file as:

hand_mouse.py

---

STEP 5: Run the Program

Open Command Prompt.

Go to the folder containing the file.

Example:

cd Desktop

or

cd Downloads

Run:

python hand_mouse.py

---

STEP 6: Controls

• Move Index Finger → Move Mouse Cursor
• Raise Index + Middle Finger → Click
• Press Q → Exit Program

---

IF PYTHON IS NOT DETECTED

Run:

python --version

If it shows an error, reinstall Python and make sure "Add Python to PATH" is checked.

---

IF A PACKAGE INSTALLATION FAILS

Run:

python -m pip install --upgrade pip

Then:

pip install opencv-python cvzone pyautogui mediapipe
