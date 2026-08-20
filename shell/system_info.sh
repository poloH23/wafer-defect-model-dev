#!/bin/bash
echo "============================"

echo "🖥  Hostname:"
hostname
echo "----------------------------"

echo "📦 Disk Usage:"
df -h /
echo "----------------------------"

echo "🧠 Memory Usage:"
free -h
echo "----------------------------"

echo "🧮 CPU Info:"
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core'
echo "----------------------------"

echo "🛠 OS Info:"
cat /etc/os-release | grep PRETTY_NAME
echo "----------------------------"

if command -v docker &> /dev/null; then
  echo "🐳 Docker Info:"
  docker system df
  echo "----------------------------"
else
  echo "🐳 Docker not installed."
  echo "----------------------------"
fi

echo "============================"
