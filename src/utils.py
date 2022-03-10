from datetime import datetime

# String Representations for Plotting
X_HAT_STR = "X\u0302"
LAMBDA_STR = "\u03BB"
ALPHA_STR = "\u03B1"
BETA_STR = "\u03B2"


def time_print(msg):
    """
Prints timestamp in format 'Year-month-day hour-minute-second and then msg'
    :param msg:
    """
    print(datetime.now().strftime('%Y-%m-%d %X'), msg)