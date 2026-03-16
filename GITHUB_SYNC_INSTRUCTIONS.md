# How to Sync Your Project with GitHub

It seems `git` is currently not installed or not working correctly on your machine's terminal. No worries! You can sync your project with GitHub in two simple ways. 

Here is exactly what you need to do:

---

## Method 1: The Easy Way (Using GitHub Desktop)

If you don't want to mess around with terminal commands, GitHub Desktop is the easiest way to sync your code.

1. **Download and Install GitHub Desktop**
   Go to [desktop.github.com](https://desktop.github.com/) and download the Windows installer.
2. **Log In**
   Open the app and log in with your GitHub account.
3. **Add Your Local Project**
   - In GitHub Desktop, go to **File > Add Local Repository...**
   - Click **Choose...** and select your project folder: `C:\Users\vedja\OneDrive\Desktop\ITDS PROJECT\project\`
   - Click **Add Repository**. (If it says "This directory does not appear to be a Git repository", click the link it gives you to "create a repository here").
4. **Publish to GitHub**
   - At the top of the app, click the blue **Publish repository** button.
   - Name your repository (e.g., `Indian-Traffic-Detection`).
   - Keep "Keep this code private" unchecked if you want others to see it.
   - Click **Publish repository**.

Your project is now synced! 

*(Note: The `.gitignore` file I created will automatically ensure the large `dataset`, `models`, and `runs` folders are safely ignored).*

---

## Method 2: The Command Line Way (Fixing Git)

If you want to use the terminal/VS Code console commands:

1. **Install Git**
   Download and install Git from: [gitforwindows.org](https://gitforwindows.org/). During the installation, simply keep clicking "Next" to accept the default settings.
2. **Restart VS Code**
   Once the installation finishes, **you must completely close VS Code** and reopen it. This ensures the terminal recognizes the new `git` command.
3. **Create the GitHub Repo Online**
   - Go to [GitHub.com](https://github.com/) and make sure you're logged in.
   - Click the **"+"** icon in the top right corner and select **New repository**.
   - Give it a name like `Indian-Traffic-Detection`.
   - **Do not** check "Add a README" or "Add .gitignore". Just click **Create repository**.
4. **Run These Commands**
   Open your VS Code terminal (ensure you are inside the `project` folder, which you should be) and paste these commands one by one:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Indian Traffic Vehicle pipeline"
   
   # IMPORTANT: Replace YOUR_USERNAME and YOUR_REPO with the actual link from step 3
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   
   git branch -M main
   git push -u origin main
   ```

---

## Method 3: The Drag-and-Drop Way (Direct Upload)
If you just want your code backed up immediately without installing anything:

1. Go to [GitHub.com](https://github.com/) and click **New repository**.
2. Name it and click **Create repository**.
3. On the next page, click standard **"upload an existing file"** link.
4. Open your File Explorer to `C:\Users\vedja\OneDrive\Desktop\ITDS PROJECT\project\`.
5. Highlight the files (`README.md`, `requirements.txt`, `.gitignore`, `run_pipeline.py`) and folders (`detection`, `enhancement`, `evaluation`, `results`, `training`, `weather_simulation`) and **drag and drop** them into the browser.
   - *Do not drag the `dataset`, `runs`, or `models` folders as they are too large.*
6. Click **Commit Changes**.
