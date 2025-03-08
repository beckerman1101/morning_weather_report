import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

today = datetime.now(ZoneInfo('America/Denver'))
todaystr = today.strftime('%m/%d')
filestr = today.strftime('%Y%m%d')



base_dir = os.path.dirname(os.path.abspath(__file__))

# Gmail SMTP settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Sender's email and App Password (for Gmail 2FA)
SENDER_EMAIL = "beckerman1101@gmail.com"
SENDER_PASSWORD = os.getenv('GMAIL_PW') # Use an App Password if 2FA is enabled

# Recipient email
RECIPIENT_EMAIL = "brendan.eckerman@state.co.us,michael.chapman@state.co.us,nicholas.barlow@state.co.us"

# Email Subject & Body
SUBJECT = f"Morning Weather Report - {todaystr}"
BODY = "Attached is today's Morning Weather Report. If you have questions or the report did not generate correctly, please reach out to Brendan at brendan.eckerman@state.co.us"

# Path to the PNG file
ATTACHMENT_PATH = f'{filestr}_MWR.png' # Update with your PNG file path

MAX_WAIT_TIME = 240  # Maximum time (in seconds) to wait for the PNG
WAIT_INTERVAL = 10  # Check every 5 seconds

start_time = time.time()

while not os.path.exists(ATTACHMENT_PATH):
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_WAIT_TIME:
        print(f"❌ PNG file did not appear within {MAX_WAIT_TIME} seconds. Exiting.")
        exit(1)
    
    print(f"Waiting for PNG file... {elapsed_time:.0f}s elapsed")
    time.sleep(WAIT_INTERVAL)  # Wait before checking again

print(f"✅ PNG file found after {time.time() - start_time:.0f}s: {ATTACHMENT_PATH}")

def send_email():
    # Create the MIME email object
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = SUBJECT
    
    # Attach the email body
    msg.attach(MIMEText(BODY, 'plain'))

    # Attach the PNG file
    try:
        with open(ATTACHMENT_PATH, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(ATTACHMENT_PATH)}')
            msg.attach(part)
    except Exception as e:
        print(f"Error attaching file: {e}")
        return

    # Send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Encrypts the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Log in using email and App Password
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Main function to call the send_email function
if __name__ == "__main__":
    send_email()
