import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
GOOGLE_EMAIL = os.getenv("GOOGLE_EMAIL")
GOOGLE_PASSWORD = os.getenv("GOOGLE_PASSWORD")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS")

today = datetime.today()
todaystr = today.strftime('%Y%m%d')
path = os.path.join(base_dir, 'daily_file', f'24houraccum_{todaystr}.png')

def send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path):
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)  # Convert list to comma-separated string
    msg['Subject'] = subject

    # Attach the body of the email
    msg.attach(MIMEText(body, 'plain'))

    # Open the attachment file in binary mode
    with open(attachment_file_path, 'rb') as attachment_file:
        # Attach the file to the email
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_file.read())
        encoders.encode_base64(part)  # Encode the attachment
        part.add_header('Content-Disposition', f'attachment; filename={attachment_file_path.split("/")[-1]}')
        msg.attach(part)

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, f"{GOOGLE_PASSWORD}")  # Use an app password if 2FA is enabled
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            print(f'Email sent successfully to: {", ".join(receiver_emails)} with attachment: {attachment_file_path}')
    except Exception as e:
        print(f"Failed to send email: {e}")

recipients = EMAIL_RECIPIENTS.split(",")
# Example usage
sender_email = f"{GOOGLE_EMAIL}"
receiver_emails = recipients  # Add multiple emails in a list
subject = "24-Hour Snowfall Report"
body = "This email is automated. Attached is the report for snowfall statewide in the past 24 hours. Data is preliminary and has not been refined for quality control."
attachment_file_path = path  # Make sure the path is correct

send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path)

print(f"GOOGLE_EMAIL: {GOOGLE_EMAIL}")  # Don't print the password for security reasons
print(f"EMAIL_RECIPIENTS: {EMAIL_RECIPIENTS}")
