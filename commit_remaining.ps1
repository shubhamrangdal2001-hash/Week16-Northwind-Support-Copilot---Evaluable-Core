git config user.email "shubham@example.com"
git config user.name "Shubham Rangdal"
# Ensure remote is correct
if (git remote | Where-Object { $_ -eq "origin" }) {
    git remote set-url origin https://github.com/shubhamrangdal2001-hash/Week16-Northwind-Support-Copilot---Evaluable-Core.git
} else {
    git remote add origin https://github.com/shubhamrangdal2001-hash/Week16-Northwind-Support-Copilot---Evaluable-Core.git
}
# Commit each top‑level folder separately, ignoring .venv, .git, and .gemini
Get-ChildItem -Directory | Where-Object { 
    $_.Name -notmatch '^\.(git|venv|gemini)$' 
} | ForEach-Object {
    $folder = $_.FullName
    git add "$folder"
    git commit -m "Add $($_.Name) folder"
}
# Finally push all commits
git push -u origin main
 