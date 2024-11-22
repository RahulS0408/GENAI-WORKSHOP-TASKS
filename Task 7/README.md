# Text-to-Speech System using pyttsx3 in Google Colab

## Description
This project demonstrates how to create a Text-to-Speech (TTS) system using the `pyttsx3` library in Google Colab. The system converts input text into speech and saves it as a `.wav` file. The generated speech can be played directly in the notebook.
## Setup

### Step 1: Install Necessary Libraries
First, install the required system packages and Python libraries:
```python
!apt-get update && apt-get install -y espeak espeak-ng
!pip install pyttsx3

---

```markdown
### Step 2: Import Libraries
Import the necessary libraries for text-to-speech conversion and audio playback:
```python
import pyttsx3
import IPython.display as ipd

---

```markdown
### Step 3: Initialize the TTS Engine
Initialize the `pyttsx3` TTS engine:
```python
engine = pyttsx3.init()

---

```markdown
### Step 4: Define the Text-to-Speech Function
Create a function to convert text to speech and save it to a file:
```python
def text_to_speech(text, filename="output.wav"):
    engine.save_to_file(text, filename)
    engine.runAndWait()

---

```markdown
### Step 5: Convert Text to Speech
Use the defined function to convert text to speech and save the output:
```python
text = "Hello Rahul how are you"
output_file = "output.wav"
text_to_speech(text, output_file)

---

```markdown
### Step 6: Play the Generated Speech
Play the generated speech file using IPython's display functionalities:
```python
ipd.Audio(output_file)
