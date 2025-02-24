import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys
import os
import codecs

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')  # For print statements
sys.stdin.reconfigure(encoding='utf-8')   # If using input
sys.stderr.reconfigure(encoding='utf-8')  # For error messages

# Force UTF-8 for environment
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"


base_dir = os.path.dirname(os.path.abspath(__file__))
email = str(os.getenv("GOOGLE_EMAIL"))
pw = str(os.getenv("GOOGLE_PASSWORD"))
recipients = str(os.getenv("EMAIL_RECIPIENTS"))

#recipients = EMAIL_RECIPIENTS.split(",") if EMAIL_RECIPIENTS else []


today = datetime.today()
todaystr = today.strftime('%Y%m%d')
path = os.path.join(base_dir, 'daily_file', f'24houraccum_{todaystr}.png')

def send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)  # Convert list to comma-separated string
    msg['Subject'] = subject

    # 🔹 FIX 1: Encode email body as UTF-8
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # Attach the file
    try:
        with open(attachment_file_path, 'rb') as attachment_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment_file.read())
            encoders.encode_base64(part)  
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_file_path)}"')
            msg.attach(part)
    except Exception as e:
        print(f"Error attaching file: {e}")

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, pw)  
            
            # 🔹 FIX 2: Encode the entire email as UTF-8 before sending
            server.sendmail(sender_email, receiver_emails, msg.as_string().encode('utf-8'))

        print(f'Email sent successfully to: {", ".join(receiver_emails)}')

    except Exception as e:
        print(f"Failed to send email: {e}")


#recipients = recipients.split(",")
# Example usage
sender_email = email
receiver_emails = recipients  # Add multiple emails in a list
subject = "24-Hour Snowfall Report"
body = "This email is automated. Attached is the report for snowfall statewide in the past 24 hours. Data is preliminary and has not been refined for quality control."
attachment_file_path = path  # Make sure the path is correct

send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path)


