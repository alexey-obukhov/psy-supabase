async function checkBackendStatus() {
  try {
    // Replace with a lightweight endpoint your backend exposes
    const res = await fetch('/health_check', { method: 'GET' });
    if (!res.ok) throw new Error();
    document.getElementById('status-message').style.display = 'none';
  } catch {
    const msg = document.getElementById('status-message');
    msg.textContent = "Sorry, the chat service is currently unavailable.\nWe're aware of the issue and will restore service as soon as possible.\nYou could contact the autor of the project to start the service\n(Please see the Contact page for more information how to reach the project author).\nThank you for your patience!";
    msg.style.display = 'block';
  }
}

// Call this on page load and optionally on send
checkBackendStatus();
