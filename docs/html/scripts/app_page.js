document.addEventListener('DOMContentLoaded', () => {
  'use strict';
  
  // --- 1. DOM Element References ---
  const chatForm = document.getElementById('chat-form');
  const usernameInput = document.getElementById('username');
  const questionTextarea = document.getElementById('question');
  const chatLog = document.getElementById('chat-log');
  const sendButton = document.getElementById('send-button');
  const processingIndicator = document.getElementById('processing-indicator');
  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');

  // API endpoint from the original script
  const API_ENDPOINT = 'http://psy-supabase.com:5008/chat';

  // --- 2. UI Enhancements ---
  
  // Mobile navigation toggle
  if (menuToggle) {
    menuToggle.addEventListener('click', function() {
      siteNav.classList.toggle('active');
      this.classList.toggle('open');
    });
  }

  // Add CSS for menu toggle animation
  const style = document.createElement('style');
  style.textContent = `
    .menu-toggle.open span:nth-child(1) {
      transform: rotate(45deg) translate(5px, 5px);
    }
    
    .menu-toggle.open span:nth-child(2) {
      opacity: 0;
    }
    
    .menu-toggle.open span:nth-child(3) {
      transform: rotate(-45deg) translate(5px, -5px);
    }
    
    .fade-in {
      opacity: 0;
      transform: translateY(20px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    
    .fade-in.appear {
      opacity: 1;
      transform: translateY(0);
    }
  `;
  document.head.appendChild(style);

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        
        // Close mobile menu if open
        if (siteNav && siteNav.classList.contains('active')) {
          siteNav.classList.remove('active');
          if (menuToggle) menuToggle.classList.remove('open');
        }
      }
    });
  });

  // Intersection Observer for animations
  const animatedElements = document.querySelectorAll('.feature-card, .process-step');
  
  if ('IntersectionObserver' in window && animatedElements.length > 0) {
    const appearOptions = {
      threshold: 0.15,
      rootMargin: '0px 0px -50px 0px'
    };
    
    const appearOnScroll = new IntersectionObserver(function(entries, observer) {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('appear');
          observer.unobserve(entry.target);
        }
      });
    }, appearOptions);
    
    animatedElements.forEach(element => {
      element.classList.add('fade-in');
      appearOnScroll.observe(element);
    });
  }

  // --- 3. Auto-resize Textarea ---
  questionTextarea.addEventListener('input', () => {
    questionTextarea.style.height = 'auto';
    questionTextarea.style.height = `${questionTextarea.scrollHeight}px`;
  });

  // --- 4. Form Submission Handler ---
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Refined validation logic
    usernameInput.classList.remove('input-error');
    questionTextarea.classList.remove('input-error');

    const username = usernameInput.value.trim();
    const question = questionTextarea.value.trim();

    if (!username ||!question) {
      if (!username) {
        usernameInput.classList.add('input-error');
      }
      if (!question) {
        questionTextarea.classList.add('input-error');
      }
      return; 
    }

    appendMessage(question, 'user');

    questionTextarea.value = '';
    questionTextarea.style.height = 'auto';

    setFormDisabled(true);
    showProcessingIndicator(true);

    try {
      // API Fetch Request
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': username
        },
        body: JSON.stringify({ question })
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMessage = data.response ||
          `An error occurred (Status: ${response.status}).`;
        appendMessage(errorMessage, 'error');
      } else {
        let botMessage = data.response ||
          'I received your message, but I have nothing to say.';

        botMessage = botMessage.replace(/\\u([\dA-F]{4})/gi, (match, grp) => {
          return String.fromCharCode(parseInt(grp, 16));
        });
        botMessage = botMessage.replace(/\n/g, '<br>');

        appendMessage(botMessage, 'bot');
      }
    } catch (err) {
      appendMessage('I\'m having a little trouble connecting right now. Please check your connection and try again.', 'error');
      console.error('Fetch Error:', err);
    } finally {
      showProcessingIndicator(false);
      setFormDisabled(false);
    }
  });

  // --- 5. Helper Functions ---

  function appendMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${type}-message`);

    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('message-bubble');
    bubbleDiv.innerHTML = `<p>${text}</p>`;

    if (type === 'error') {
      bubbleDiv.style.backgroundColor = 'var(--color-error)';
      bubbleDiv.style.color = 'var(--color-white)';
    }
    
    messageDiv.appendChild(bubbleDiv);
    chatLog.appendChild(messageDiv);

    scrollToBottom();
  }

  function showProcessingIndicator(isVisible) {
    processingIndicator.style.display = isVisible? 'flex' : 'none';
    if (isVisible) {
      scrollToBottom();
    }
  }

  function setFormDisabled(isDisabled) {
    questionTextarea.disabled = isDisabled;
    sendButton.disabled = isDisabled;
    usernameInput.disabled = isDisabled;
  }

  function scrollToBottom() {
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // --- 6. Initialize the page ---

  // Add welcome message if chat is empty
  if (chatLog && chatLog.children.length === 0) {
    appendMessage('Hello. I\'m your Mindful Assistant. How can I support you today?', 'bot');
  }
});

const tabBtns = document.querySelectorAll('.tab-btn');
const advantageCards = document.querySelectorAll('.advantage-card');

if (tabBtns.length > 0) {
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active class from all buttons
      tabBtns.forEach(b => b.classList.remove('active'));
      
      // Add active class to clicked button
      btn.classList.add('active');
      
      const category = btn.getAttribute('data-tab');
      
      // Show/hide cards based on category
      advantageCards.forEach(card => {
        if (category === 'all' || card.getAttribute('data-category') === category) {
          card.style.display = 'block';
          setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
          }, 10);
        } else {
          card.style.opacity = '0';
          card.style.transform = 'translateY(10px)';
          setTimeout(() => {
            card.style.display = 'none';
          }, 300);
        }
      });
    });
  });
}