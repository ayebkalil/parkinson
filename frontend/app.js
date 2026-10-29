/**
 * NEURO-PREDICT HUD & GLASSMORPHIC AUTHENTICATION CONTROLLER
 * - Seamless ping-pong background video loop
 * - Clean first page with crystal-clear 2.5K video
 * - Click anywhere to summon the frosted glass Login & Sign Up modal
 * - Interactive form tabs, role chips, demo autofill, and password toggle
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const bgVideo = document.getElementById('bgVideo');
  const glassModalBackdrop = document.getElementById('glassModalBackdrop');
  const glassCard = document.getElementById('glassCard');
  const modalCloseBtn = document.getElementById('modalCloseBtn');
  
  const tabLogin = document.getElementById('tabLogin');
  const tabSignup = document.getElementById('tabSignup');
  const tabSwitch = document.querySelector('.tab-switch');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');
  
  const hudToast = document.getElementById('hudToast');
  const toastTitle = document.getElementById('toastTitle');
  const toastMessage = document.getElementById('toastMessage');

  // -------------------------------------------------------------
  // Video Autoplay Guarantee
  // -------------------------------------------------------------
  if (bgVideo) {
    bgVideo.play().catch(() => {
      // If browser policy requires gesture, trigger on first click
      const resumePlay = () => {
        bgVideo.play();
        window.removeEventListener('click', resumePlay);
      };
      window.addEventListener('click', resumePlay);
    });
  }

  // -------------------------------------------------------------
  // Click-Anywhere Action -> Reveal Glassmorphism Form
  // -------------------------------------------------------------
  let isModalOpen = false;

  function openModal() {
    if (isModalOpen) return;
    isModalOpen = true;
    document.body.classList.add('modal-active');
    
    // Auto-focus first input in the active form
    setTimeout(() => {
      const activeInput = document.querySelector('.auth-form.active input');
      if (activeInput) activeInput.focus();
    }, 250);
  }

  function closeModal() {
    if (!isModalOpen) return;
    isModalOpen = false;
    document.body.classList.remove('modal-active');
  }

  // Detect clicks anywhere on the document
  document.addEventListener('click', (e) => {
    if (!isModalOpen) {
      // First click anywhere opens the glass form
      openModal();
    } else {
      // When modal is open, clicking backdrop outside the glass card closes it
      if (glassModalBackdrop.contains(e.target) && !glassCard.contains(e.target)) {
        closeModal();
      }
    }
  });

  // Keyboard navigation: Enter/Space opens, ESC closes
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isModalOpen) {
      closeModal();
    } else if ((e.key === 'Enter' || e.key === ' ') && !isModalOpen) {
      openModal();
    }
  });

  // Modal Close Button
  modalCloseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    closeModal();
  });

  // Prevent clicks inside the glass card from closing the modal
  glassCard.addEventListener('click', (e) => {
    e.stopPropagation();
  });

  // -------------------------------------------------------------
  // Tab Switching (Sign In vs Create Account)
  // -------------------------------------------------------------
  function switchTab(target) {
    if (target === 'loginForm') {
      tabLogin.classList.add('active');
      tabSignup.classList.remove('active');
      tabSwitch.removeAttribute('data-active');
      loginForm.classList.add('active');
      signupForm.classList.remove('active');
    } else {
      tabLogin.classList.remove('active');
      tabSignup.classList.add('active');
      tabSwitch.setAttribute('data-active', 'signup');
      loginForm.classList.remove('active');
      signupForm.classList.add('active');
    }
  }

  tabLogin.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('loginForm');
  });

  tabSignup.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('signupForm');
  });

  // -------------------------------------------------------------
  // Role Selector Chips
  // -------------------------------------------------------------
  const roleChips = document.querySelectorAll('.role-chip');
  roleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      roleChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
    });
  });

  // -------------------------------------------------------------
  // Password Visibility Toggle
  // -------------------------------------------------------------
  const passToggleButtons = document.querySelectorAll('.password-toggle-btn');
  passToggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const input = btn.parentElement.querySelector('input');
      const eyeOpen = btn.querySelector('.eye-open');
      const eyeClosed = btn.querySelector('.eye-closed');
      
      if (input.type === 'password') {
        input.type = 'text';
        eyeOpen.classList.add('hidden');
        eyeClosed.classList.remove('hidden');
      } else {
        input.type = 'password';
        eyeOpen.classList.remove('hidden');
        eyeClosed.classList.add('hidden');
      }
    });
  });

  // -------------------------------------------------------------
  // -------------------------------------------------------------
  // Live Terminal Chrono / UTC Clock for Top-Right HUD Module
  // -------------------------------------------------------------
  const hudClock = document.getElementById('hudClock');
  function updateHudClock() {
    if (!hudClock) return;
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const hours = pad(now.getUTCHours());
    const minutes = pad(now.getUTCMinutes());
    const seconds = pad(now.getUTCSeconds());
    hudClock.textContent = `${hours}:${minutes}:${seconds} UTC`;
  }
  updateHudClock();
  setInterval(updateHudClock, 1000);

  // -------------------------------------------------------------
  // Quick Clinician Demo Profile Chips
  // -------------------------------------------------------------
  const demoChips = document.querySelectorAll('.demo-chip');
  demoChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.preventDefault();
      const email = chip.getAttribute('data-email');
      const role = chip.getAttribute('data-role');
      const name = chip.getAttribute('data-name');
      
      const doctorNameInput = document.getElementById('loginDoctorName');
      if (doctorNameInput && name) {
        doctorNameInput.value = name;
      }

      document.getElementById('loginEmail').value = email;
      document.getElementById('loginPassword').value = 'PPMI-Neural-2026!';
      
      roleChips.forEach(c => {
        const input = c.querySelector('input');
        if (input.value === role) {
          c.classList.add('active');
          input.checked = true;
        } else {
          c.classList.remove('active');
        }
      });
      
      showToast('PROFILE LOADED', `Credentials for ${name || chip.textContent.trim()} loaded.`);
    });
  });

  // -------------------------------------------------------------
  // Interactive Feedback Toast
  // -------------------------------------------------------------
  let toastTimeout = null;
  function showToast(title, message, duration = 3800) {
    const toastTitleEl = document.getElementById('toastTitle');
    const toastMessageEl = document.getElementById('toastMessage');
    const hudToastEl = document.getElementById('hudToast');
    
    if (toastTitleEl) toastTitleEl.textContent = title;
    if (toastMessageEl) toastMessageEl.textContent = message;
    if (hudToastEl) hudToastEl.classList.add('show');
    
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      if (hudToastEl) hudToastEl.classList.remove('show');
    }, duration);
  }

  // -------------------------------------------------------------
  // High-Tech Portal Transition & Audio Pulse
  // -------------------------------------------------------------
  const portalTransitionOverlay = document.getElementById('portalTransitionOverlay');
  const warpDoctorTitle = document.getElementById('warpDoctorTitle');

  function playWarpSound() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(320, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(740, ctx.currentTime + 0.6);
      
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.75);
      
      osc.connect(gain);
      gain.connect(ctx.destination);
      
      osc.start();
      osc.stop(ctx.currentTime + 0.75);
    } catch (e) {
      // Audio policy catch
    }
  }

  function triggerPortalWarp(doctorName, role, email) {
    // 1. Save practitioner credentials to localStorage for dashboard
    try {
      const profile = {
        name: doctorName || 'Dr. Liza Vance, MD',
        role: role || 'Chief Neurologist',
        email: email || 'dr.liza@neuro-ppmi.org',
        timestamp: new Date().toISOString()
      };
      localStorage.setItem('medix_active_doctor', JSON.stringify(profile));
    } catch (err) {
      console.warn('[MEDIX] Storage notice:', err);
    }

    // 2. Animate visual portal overlay & sound
    try {
      const warpTitleEl = document.getElementById('warpDoctorTitle');
      if (warpTitleEl) {
        warpTitleEl.textContent = `WELCOME, ${(doctorName || 'CLINICIAN').toUpperCase()}`;
      }
      playWarpSound();
      const overlay = document.getElementById('portalTransitionOverlay');
      if (overlay) {
        overlay.classList.add('active');
      }
    } catch (err) {
      console.warn('[MEDIX] Transition notice:', err);
    }

    // 3. Guaranteed smooth redirect to the clinical dashboard
    setTimeout(() => {
      window.location.assign('dashboard.html');
    }, 950);
  }

  // -------------------------------------------------------------
  // Form Submission Handlers
  // -------------------------------------------------------------
  loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    let doctorName = 'Dr. Liza Vance, MD';
    const doctorNameInput = document.getElementById('loginDoctorName');
    if (doctorNameInput && doctorNameInput.value.trim()) {
      doctorName = doctorNameInput.value.trim();
    }
    
    let email = 'dr.liza@neuro-ppmi.org';
    const emailInput = document.getElementById('loginEmail');
    if (emailInput && emailInput.value.trim()) {
      email = emailInput.value.trim();
    }
    
    const activeChip = document.querySelector('.role-chip.active .chip-text');
    const roleText = activeChip ? activeChip.textContent.trim() : 'Chief Neurologist';
    
    const submitBtn = document.getElementById('loginSubmitBtn');
    if (submitBtn) {
      submitBtn.disabled = true;
      const btnText = submitBtn.querySelector('.btn-text');
      if (btnText) btnText.textContent = 'AUTHENTICATING...';
    }
    
    showToast(
      'ACCESS AUTHORIZED',
      `Welcome, ${doctorName}. Initializing Parkinson's Decision Support Suite...`
    );

    setTimeout(() => {
      triggerPortalWarp(doctorName, roleText, email);
    }, 450);
  });

  signupForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    let name = 'Dr. Michael Vance, MD';
    const nameInput = document.getElementById('signupName');
    if (nameInput && nameInput.value.trim()) {
      name = nameInput.value.trim();
    }
    
    let email = 'practitioner@neuro-ppmi.org';
    const emailInput = document.getElementById('signupEmail');
    if (emailInput && emailInput.value.trim()) {
      email = emailInput.value.trim();
    }
    
    const submitBtn = document.getElementById('signupSubmitBtn');
    if (submitBtn) {
      submitBtn.disabled = true;
      const btnText = submitBtn.querySelector('.btn-text');
      if (btnText) btnText.textContent = 'ACTIVATING CREDENTIALS...';
    }
    
    showToast(
      'ACCOUNT CREATED',
      `Practitioner profile registered for ${name}. Loading MEDIX Workspace...`
    );

    setTimeout(() => {
      triggerPortalWarp(name, 'Neurologist', email);
    }, 550);
  });

  // Forgot Password trigger
  const forgotPasswordLink = document.getElementById('forgotPasswordLink');
  if (forgotPasswordLink) {
    forgotPasswordLink.addEventListener('click', (e) => {
      e.preventDefault();
      showToast(
        'SECURITY OVERRIDE',
        'Passkey reset dispatch sent to institutional biometric authenticator.'
      );
    });
  }

});
