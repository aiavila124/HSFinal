// Obtener referencias a los elementos del DOM
const openModalButton = document.getElementById('open-modal');
const closeModalButton = document.getElementById('close-modal');
const modalContainer = document.getElementById('modal-container');
const errorMessage = document.getElementById('error-message');

// Función para abrir la ventana emergente
function openModal(message) {
  errorMessage.textContent = message;
  modalContainer.classList.add('open');
}

// Función para cerrar la ventana emergente
function closeModal() {
  modalContainer.classList.remove('open');
}

// Agregar eventos a los botones
openModalButton.addEventListener('click', openModal);
closeModalButton.addEventListener('click', closeModal);
