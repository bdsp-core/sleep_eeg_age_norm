The `age_dependent_sleep_eeg_spectrum.py` script makes a batch of
images showing the spectrograms and power vs frequency for a
specific queried age (with standard deviations).
To run this script simply execute `python
age_dependent_sleep_eeg_spectrum.py`.
(Note: if you do not have the necessary libraries please install
them via `pip`.)

The `alek_prototype.ipynb` script, gives a demo of some of the
ways to make a jupyter notebook into an interactive website.
To run this, I recommend the following:

- `python3 -m virtualenv venv` (create virtual environment
- `. ./venv/bin/activate` (activate the virtual environment)
    named *venv*)
- `pip install -r requirements.txt` (installs the requirements)
- `viola alek_prototype.ipynb` (this will open a locally hosted
website displying the app)
- try using the dropdown menu to view different images

Note that there is code for generating all of the images, but
only some of them are displayed on the website.

**This is a work in progress!** Let me know if you have any questions.
