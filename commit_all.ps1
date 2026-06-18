git config user.email "shubham@example.com"
git config user.name "Shubham Rangdal"
if (git remote | Where-Object { $_ -eq "origin" }) { git remote remove origin }
git remote add origin https://github.com/shubhamrangdal2001-hash/Week16-Northwind-Support-Copilot---Evaluable-Core.git
git checkout -B main
Get-ChildItem -Recurse -File | ForEach-Object {
    git add $_.FullName
    git commit -m "Add $($_.Name)"
}
git push -u origin main
