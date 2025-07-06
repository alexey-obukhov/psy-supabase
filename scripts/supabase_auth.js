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
const SUPABASE_URL = 'https://supabase.psy-supabase.com';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBmY2Rpbmp4and6Y2R4Z3Z4bHVkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzYzMzc3MTYsImV4cCI6MjA1MTkxMzcxNn0.n22JvFNTRvKZnKDl7YGGXjc9Y6HaRcaap9pM-N2eZrs';

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
