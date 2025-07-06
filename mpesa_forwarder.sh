#!/data/data/com.termux/files/usr/bin/bash

# --- Configuration ---
RECIPIENT_EMAIL="mungairamsey01@gmail.com"
LAST_MSG_CONTENT_FILE="/data/data/com.termux/files/home/storage/learning/learning/last_mpesa_content.txt"

# --- Main Script ---

# Create the last message content file if it doesn't exist
touch "$LAST_MSG_CONTENT_FILE"

# Get the content of the last successfully forwarded message
LAST_SENT_CONTENT=$(cat "$LAST_MSG_CONTENT_FILE")

# Get the content of the latest notification containing "MPESA"
# We use grep to find the line, and head -n 1 to get the most recent one
LATEST_CONTENT=$(termux-notification-list | grep "MPESA" | head -n 1)

# If no message is found, exit
if [[ -z "$LATEST_CONTENT" ]]; then
    exit 0
fi

# If the latest message is the same as the last one we sent, exit
if [[ "$LATEST_CONTENT" == "$LAST_SENT_CONTENT" ]]; then
    exit 0
fi

# --- Send Email ---
SUBJECT="New M-Pesa Transaction"
EMAIL_BODY="$LATEST_CONTENT"

# Pipe the email content to msmtp
printf "Subject: %s\n\n%s" "$SUBJECT" "$EMAIL_BODY" | msmtp -C /data/data/com.termux/files/home/storage/learning/learning/.msmtprc "$RECIPIENT_EMAIL"

# --- Save Last Message Content ---
# If the email was sent successfully, save the new message content
if [ $? -eq 0 ]; then
    echo "$LATEST_CONTENT" > "$LAST_MSG_CONTENT_FILE"
fi