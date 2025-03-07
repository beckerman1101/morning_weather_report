import requests
import os
from github import Github

# GitHub Configuration - Use environment variables instead of hardcoding credentials
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Set this in your environment
REPO_NAME = 'beckerman1101/daily_accum_mapping'  # Replace with your repository
BRANCH_NAME = 'main'
FILE_PATH_IN_REPO = 'daily_file/filename.tar.gz'
DOWNLOAD_URL = 'https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz'

# Download the file from the URL
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download file: {response.status_code}")

# Push the file to GitHub repository
def push_to_github(file_path):
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

    try:
        # Read file content
        with open(file_path, 'rb') as file_content:
            content = file_content.read()

        # Commit file to GitHub
        repo.create_file(FILE_PATH_IN_REPO, "Add downloaded file", content, branch=BRANCH_NAME)
        print(f"File successfully pushed to GitHub at {REPO_NAME}/{BRANCH_NAME}")
    except Exception as e:
        print(f"Error pushing file to GitHub: {e}")

# Main function
def main():
    downloaded_filename = 'downloaded_file.tar.gz'
    download_file(DOWNLOAD_URL, downloaded_filename)
    push_to_github(downloaded_filename)
    os.remove(downloaded_filename)

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN is not set. Set it as an environment variable.")
    else:
        main()
