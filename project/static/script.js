// Show/hide password functionality for login page
document.addEventListener('DOMContentLoaded', function() {
    const passwordInput = document.getElementById('password');
    const showPasswordCheckbox = document.getElementById('showPassword');

    if (passwordInput && showPasswordCheckbox) {
        showPasswordCheckbox.addEventListener('change', function() {
            console.log('Checkbox changed:', this.checked);
            passwordInput.type = this.checked ? 'text' : 'password';
        });
    }
});

// Modal functionality
const modal = document.getElementById('editProfileModal');
const editBtn = document.getElementById('editProfileBtn');
const closeBtn = document.getElementById('closeModal');
const cancelBtn = document.getElementById('cancelEdit');

if (editBtn) {
    editBtn.addEventListener('click', () => {
        modal.classList.remove('hidden');
    });
}

if (closeBtn) {
    closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });
}

if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });
}

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.add('hidden');
    }
});