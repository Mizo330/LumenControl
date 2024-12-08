
# LumenControl  

This repository contains the code for my BSc thesis. The project analyzes audio input, generates OSC signals using AI models and standard signal processing techniques commonly applied in Music Information Retrieval (MIR), and sends these signals to a light control program (e.g., QLC+). The program can control DMX fixtures via a virtual console.

> **Note:** This project is a work in progress. Models used in the code are subject to change.
>
> Currently, the project utilizes the [Beat This!](https://github.com/CPJKU/beat_this.git) beat recognition model.

This project draws inspiration from scheb's [sound_to_light_osc](https://github.com/scheb/sound-to-light-osc.git) project. The core structure of the code is adapted from their implementation.

---

## Installation - Python Code

1. Install Visual Studio Code (VSCode).
2. Install the Docker extension for VSCode.
3. Run the following command to build the Docker image:
   ```
   ./build_docker
   ```
4. Start the Docker container:
   ```
   ./run_docker
   ```
5. In the Docker extension tab within VSCode, right-click the container and select "Attach Visual Studio Code."
6. Execute the `/lumen/lumenControl.py` file.

### Notes:
- Thresholds, such as RMS values, may need adjustment based on your audio input.

---

## Installation - Light Control

To control DMX fixtures, you must install QLC+ and complete some setup tasks. These steps are simplified for convenience; it is recommended to research how OSC and QLC+ work for more detailed guidance.

1. Assign a generic OSC control to the desired universe in the "Profiles" tab under the "Input/Output" section.
2. Configure the OSC control and map the channels used by the code. The easiest method is to enable the assignment wizard, run the Python code while playing audio, and allow the wizard to detect all inputs.
3. Create your own fixture setup and map the OSC signals to match your requirements.

---

## TODOs:
- [ ] Finalize README
- [ ] Implement dynamic thresholding
- [ ] Fix issues with OSC signal handling
- [ ] Develop a better solution for managing build-ups
- [ ] Identify or create a real-time segment labeling neural network model
- [ ] Replace QLC+ with direct DMX communication

