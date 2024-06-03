from time import sleep
import sys

response = "This is a response"

# Store the number of chunks to erase later
num_chunks = len(response)

for chunk in response:
    print(chunk, end=" ", flush=True)
    sleep(1)

# Move the cursor up and clear the lines
for _ in range(num_chunks):
    sys.stdout.write('\033[F\033[K')

# Print the full response
print(response)
