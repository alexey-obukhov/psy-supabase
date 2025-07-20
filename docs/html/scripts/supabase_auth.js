// supabase_auth.js
// Shared Supabase authentication logic for all pages

// Load Supabase UMD library if not already loaded
(function loadSupabaseUMD() {
  if (!window.supabase) {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.42.7/dist/umd/supabase.min.js';
    script.async = false;
    document.head.appendChild(script);
  }
})();

// Initialize Supabase client (global)
const SUPABASE_URL = 'http://192.168.2.150:8000'; // Your self-hosted Supabase instance
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicmVmIjoibG9jYWxob3N0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI4Njc0NTYsImV4cCI6MjA2ODIyNzQ1Nn0.OilBLadMfjrrqFSofdDOKlin0j5p4qpyu6fF_ZqFeaw';

// Use the global window.supabase from the UMD build
const supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

// Get the current authenticated user (returns a Promise)
async function getCurrentUser() {
  const { data: { user } } = await supabaseClient.auth.getUser();
  return user;
}

// Sign in with email and password (returns {data, error})
async function signIn(email, password) {
  return await supabaseClient.auth.signInWithPassword({ email, password });
}

// Sign up with email and password (returns {data, error})
async function signUp(email, password) {
  return await supabaseClient.auth.signUp({ email, password });
}

// Sign out (returns {error})
async function signOut() {
  return await supabaseClient.auth.signOut();
}

// Export for use in inline scripts
window.supabaseClient = supabaseClient;
window.getCurrentUser = getCurrentUser;
window.signIn = signIn;
window.signUp = signUp;
window.signOut = signOut;
