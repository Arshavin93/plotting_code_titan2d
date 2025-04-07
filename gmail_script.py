import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt

# Set up the SMTP server and login credentials
smtp_server = "smtp.gmail.com"
port = 587
username = "vishals.rec@gmail.com"
password = "xtev opge mmyw dubt"

recipient_email = "d22051@students.iitmandi.ac.in"
# Create a new email message
msg = MIMEMultipart("alternative")
msg["Subject"] = "Python script completed"
msg["From"] = username
msg["To"] = recipient_email

# Add the plot image to the email message
img = plt.imread("/home/ashish/Desktop/param_himalaya/pile_height_vs_time.png")
with open("/home/ashish/Desktop/param_himalaya/pile_height_vs_time.png", "rb") as fp:
    img_data = fp.read()
attachment = MIMEImage(img_data, name="pile_height_vs_time.png")
msg.attach(attachment)

# Set the sender and recipient email addresses
sender_email = username
reci_email = recipient_email

# Send the email
smtp = smtplib.SMTP(smtp_server, port)
smtp.starttls()
smtp.login(username, password)
smtp.sendmail(sender_email, recipient_email, msg.as_string())
smtp.quit()
