git config user.email "shubham@example.com"
git config user.name "Shubham Rangdal"

# Ensure remote points to the correct GitHub repository
if (git remote | Where-Object { $_ -eq "origin" }) {
    git remote set-url origin https://github.com/shubhamrangdal2001-hash/Week16-Northwind-Support-Copilot---Evaluable-Core.git
} else {
    git remote add origin https://github.com/shubhamrangdal2001-hash/Week16-Northwind-Support-Copilot---Evaluable-Core.git
}

# Commit each file with a unique message, ignoring .venv, .git, and .gemini folders
Get-ChildItem -Recurse -File | Where-Object { 
    $_.FullName -notmatch '\\\.(venv|git|gemini)\\' 
} | ForEach-Object {
    git add $_.FullName
    git commit -m "Add $($_.Name)"
}

# Push to GitHub
git push -u origin main
 