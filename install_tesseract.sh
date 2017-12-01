#!/usr/bin/env bash

echo "Installation Script for Tesseract"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
        echo "Begining to install Tesseract package"
        sudo apt-get install tesseract-ocr
        which -s pip
        if [[ $? != 0 ]] ; then
          echo "Installing Pip"
          sudo easy_install pip
        else
          echo "Pip Detected"
        fi
        pip install pytesseract
elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        # Check for Homebrew,
        # Install if we don't have it
        if test ! $(which brew); then
          echo "Installing homebrew..."
          ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
        fi
        # Ensure `homebrew` is up-to-date and ready
        echo "Updating homebrew..."
        brew doctor

        # Install Tesseract devel with English
        echo "Installing tesseract..."
        brew install tesseract 
else
        # Unknown.
        echo "This script supports only POSIX, LINUX, MacOS"
        echo "Exiting script"
        exit 1
fi
