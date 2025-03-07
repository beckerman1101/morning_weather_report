import os
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment
from base64 import b64encode
from datetime import datetime, timedelta

today = datetime.now(ZoneInfo('America/Denver'))
todaystr = today.strftime('%m/%d')
filestr = today.strftime('%Y%m%d')

base_dir = os.path.dirname(os.path.abspath(__file__))
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
FROM_EMAIL = 'brendan.eckerman@state.co.us'
TO_EMAIL = 'beckerman1101@gmail.com'
SUBJECT = 'Daily Weather Report'
BODY = 'Please find the attached weather report in PNG format.'

def send_email_with_attachment(png_filename):
    # Create SendGrid client
    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)

    # Create email components
    from_email = Email(FROM_EMAIL)
    to_email = To(TO_EMAIL)
    subject = SUBJECT
    content = Content("text/plain", BODY)

    # Create the email message
    mail = Mail(from_email, to_email, subject, content)

    # Attach the PNG file
    with open(png_filename, 'rb') as f:
        attachment = Attachment()
        attachment.content = b64encode(f.read()).decode()  # Base64 encode the content
        attachment.type = 'image/png'  # Specify the MIME type
        attachment.filename = os.path.basename(png_filename)  # Use the filename as attachment name
        attachment.disposition = 'attachment'  # Set the disposition to 'attachment'
        mail.attachment = attachment

    # Send the email
    try:
        response = sg.send(mail)
        print(f"Email sent! Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Example usage in the script
if __name__ == "__main__":
    png_filename = os.path.join(base_dir, f'{filestr}_MWR.png') # Replace with the actual path to your PNG
    send_email_with_attachment(png_filename)
