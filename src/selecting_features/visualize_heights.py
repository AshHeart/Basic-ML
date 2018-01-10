'''Here we get our first good look at one of the tools that matplotlib gives us to visualize features to help us better select said features'''
import numpy as np
import matplotlib.pyplot as plt

#The data with 500 greyhounds and 500 labs
greyhounds = 500
labs = 500

#Set our heights and somewhat randomize it
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

#View our data using a histogram
plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()
