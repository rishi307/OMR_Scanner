# OMR Scanner for Shaastra Ignite

![Shaastra-logo](./shaastra.jpeg)

## Description

Script to evaluate OMR's this Shaastra Ignite.

Outputs your marked answers.
______

## Dependencies

Opencv3

Python 2.5+, 3.4+

______
## Usage

```
python omr.py input_image
```

All intermediaries will be visible as separate panes.

Use any key to exit.

______
## Methodology

* Firstly we obtain a perspective transform, since we have a layout of the OMR sheet at hand

* Proceeding this we isolate areas where OMR bubbles are marked. This is done via edge detection and filter usage.

* Lastly, we create maps to isolate bubbles row wise, and compare their thresholded pixel values.
______
## Future Plans

1. Add CRNN integration to detect name, roll no..etc
