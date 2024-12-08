# LumenControl  

This repo is the code for my BSC thesis, that analyzes input audi music, and generates OSC signals using AI models and standard signal processing used in MIR(music information retrieval). Then it sends OSC signals for a light controller program (such as QLC+) that can control DMX fixtures with the virtual console.

>This project is still a work in progress! Models used in this code may change.
>
>Currently using the [Beat This!](https://github.com/CPJKU/beat_this.git) beat recognizer model.

I have took inspiration form scheb's [sound_to_light_osc](https://github.com/scheb/sound-to-light-osc.git) project, the backbone structure follows their code.

## Installation - Python Code

1. Install VSCode
2. Install docker extension
3. ``./build_docker``
4. ``./run_docker``
5. In the docker extension tab right click the container -> "attach visual studio code"
6. Run the /lumen/lumenControl.py file

Notes:
- Some threshold may need change to work based on your audio input, such as the RMS values. 

 ## Installation - Light Control

To actually control DMX fixtures you need to install QLC+ and do some setup beforehand. These steps are simplified, so I advise you to look up how OSC and QLC works.

 1.  Assign a generic osc control to your used universe in the profiles tab under input/output section.
 2. Configure the osc control and assign the channels that the code uses. The easiest way is to turn on the assign wizard, run the python code (also play some music) and let the wizard listen for all the inputs.
 3. Make your own fixture setup and assign the osc signals to your liking.

## TODOs:
- [] README finalization
- [] Dynamic thresholding
- [] Fixing OSC signals mishaps
- [] Find a better solution for build-ups
- [] Find (or make) a real-time segment label nn model
- [] Ditch QLC+ and communicate using DMX directly


