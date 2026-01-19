document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded. Script is running.');

    // --- 스크립트 생성 흐름의 PPT 업로드 기능 관련 요소 (script_generator.html에만 존재) ---
    const chooseFilesBtnScript = document.querySelector('.choose-files-btn-script');
    const fileInputScript = document.getElementById('file-input-script');
    const uploadFormScript = document.getElementById('upload-ppt-form');
    const loadingMessageScript = document.getElementById('loading-message-script');
    const uploadAreaScript = document.querySelector('#upload-ppt-form .upload-area');

    // --- 대본 녹음 파일 업로드 페이지 관련 요소 (record_script.html에만 존재) ---
    const chooseAudioBtn = document.querySelector('.choose-audio-btn');
    const audioFileInput = document.getElementById('audio-file-input');
    const uploadAudioForm = document.getElementById('upload-audio-form');
    const loadingMessageAudio = document.getElementById('loading-message-audio');
    const uploadAudioArea = document.querySelector('.upload-audio-area');


    // 디버깅: 요소가 제대로 찾아졌는지 확인
    if (chooseFilesBtnScript) console.log('chooseFilesBtnScript found.');
    if (fileInputScript) console.log('fileInputScript found.');
    if (uploadFormScript) console.log('uploadFormScript found.');
    if (loadingMessageScript) console.log('loadingMessageScript found.');
    if (uploadAreaScript) console.log('uploadAreaScript found.');

    if (chooseAudioBtn) console.log('chooseAudioBtn found.');
    if (audioFileInput) console.log('audioFileInput found.');
    if (uploadAudioForm) console.log('uploadAudioForm found.');
    if (loadingMessageAudio) console.log('loadingMessageAudio found.');
    if (uploadAudioArea) console.log('uploadAudioArea found.');


    // --- 스크립트 생성 흐름의 PPT 업로드 기능 (script_generator.html) ---
    if (chooseFilesBtnScript && fileInputScript && uploadFormScript && loadingMessageScript && uploadAreaScript) {
        chooseFilesBtnScript.addEventListener('click', () => {
            fileInputScript.click();
        });

        fileInputScript.addEventListener('change', () => {
            if (fileInputScript.files.length > 0) {
                handleFileUpload(fileInputScript.files[0], uploadFormScript, loadingMessageScript);
            }
        });

        uploadAreaScript.addEventListener('dragover', (event) => {
            event.preventDefault();
            uploadAreaScript.classList.add('drag-over');
        });
        uploadAreaScript.addEventListener('dragleave', () => {
            uploadAreaScript.classList.remove('drag-over');
        });
        uploadAreaScript.addEventListener('drop', (event) => {
            event.preventDefault();
            uploadAreaScript.classList.remove('drag-over');
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0], uploadFormScript, loadingMessageScript);
            }
        });
    }

    // --- 대본 녹음 파일 업로드 기능 (record_script.html) ---
    if (chooseAudioBtn && audioFileInput && uploadAudioForm && loadingMessageAudio && uploadAudioArea) {
        chooseAudioBtn.addEventListener('click', () => {
            audioFileInput.click();
        });

        audioFileInput.addEventListener('change', () => {
            if (audioFileInput.files.length > 0) {
                handleFileUpload(audioFileInput.files[0], uploadAudioForm, loadingMessageAudio);
            }
        });

        uploadAudioArea.addEventListener('dragover', (event) => {
            event.preventDefault();
            uploadAudioArea.classList.add('drag-over');
        });
        uploadAudioArea.addEventListener('dragleave', () => {
            uploadAudioArea.classList.remove('drag-over');
        });
        uploadAudioArea.addEventListener('drop', (event) => {
            event.preventDefault();
            uploadAudioArea.classList.remove('drag-over');
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0], uploadAudioForm, loadingMessageAudio);
            }
        });
    }


    // 파일 업로드 및 서버 통신을 처리하는 공통 함수
    function handleFileUpload(file, formElement, loadingMessageElement) {
        console.log('handleFileUpload called with file:', file.name, 'for form:', formElement.id);
        const formData = new FormData();
        // 파일 input의 name 속성에 따라 'file' 또는 'audio_data'로 append
        formData.append(file.type.startsWith('audio') ? 'audio_data' : 'file', file);

        if (loadingMessageElement) {
            loadingMessageElement.style.display = 'block';
        }

        console.log('Sending fetch request to:', formElement.action);

        fetch(formElement.action, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Fetch response received. Status:', response.status, 'Redirected:', response.redirected, 'URL:', response.url);
            // Flask의 flash 메시지를 클라이언트에서 처리하기 위한 로직 추가
            if (!response.redirected && response.status === 200) { // 리다이렉트가 아닌 200 OK 응답일 경우 (주로 flash 메시지 포함)
                return response.text().then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const flashMessages = doc.querySelector('.flash-messages');
                    if (flashMessages) {
                        // 현재 페이지의 flash-messages 영역을 업데이트
                        const currentFlashArea = document.querySelector('.flash-messages');
                        if (currentFlashArea) {
                            currentFlashArea.innerHTML = flashMessages.innerHTML;
                            // 스크롤을 맨 위로 올려 메시지가 보이도록
                            window.scrollTo(0, 0);
                        } else {
                            // flash-messages 영역이 없으면 alert으로라도 표시
                            const messages = Array.from(flashMessages.querySelectorAll('li')).map(li => li.textContent).join('\n');
                            if (messages) alert(messages);
                        }
                    }
                    // 파일 업로드 실패 시에도 로딩 메시지 숨기기
                    if (loadingMessageElement) {
                        loadingMessageElement.style.display = 'none';
                    }
                });
            } else if (response.redirected) {
                window.location.href = response.url;
            } else if (response.headers.get('Content-Type') && response.headers.get('Content-Type').includes('application/json')) {
                return response.json().then(data => {
                    console.error('Server responded with JSON (non-redirect):', data);
                    alert(data.message || '파일 처리 중 알 수 없는 오류가 발생했습니다.');
                    window.location.reload();
                });
            } else {
                return response.text().then(text => {
                    console.error('Unexpected server response (neither redirect nor JSON/file):', text);
                    alert('파일 처리 중 예상치 못한 응답이 발생했습니다. 자세한 내용은 콘솔을 확인해주세요.');
                    window.location.reload();
                });
            }
        })
        .catch(error => {
            console.error('파일 업로드 및 처리 중 JavaScript 오류:', error);
            alert('파일 처리 중 오류가 발생했습니다.');
            window.location.reload();
        })
        .finally(() => {
            if (loadingMessageElement) {
                loadingMessageElement.style.display = 'none';
            }
        });
    }

    // --- 드롭다운 메뉴 기능 (기존 코드 유지) ---
    const toolsNavLink = document.querySelector('.main-nav li a');
    const dropdownMenu = document.querySelector('.main-nav .dropdown-menu');

    if (toolsNavLink && dropdownMenu) {
        toolsNavLink.addEventListener('mouseenter', () => {
            dropdownMenu.style.display = 'block';
        });

        toolsNavLink.parentNode.addEventListener('mouseleave', () => {
            dropdownMenu.style.display = 'none';
        });
    }

    // --- 키워드 선택 페이지 (keyword_select.html) - 선택 시 스타일 변경 ---
    const keywordItems = document.querySelectorAll('.keyword-item');
    if (keywordItems.length > 0) {
        keywordItems.forEach(item => {
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (checkbox) {
                // 페이지 로드 시 체크박스 상태에 따라 'selected' 클래스 적용
                if (checkbox.checked) {
                    item.classList.add('selected');
                }

                item.addEventListener('click', (event) => {
                    // label 클릭 시 체크박스 상태 토글
                    checkbox.checked = !checkbox.checked;
                    // 'selected' 클래스 토글
                    item.classList.toggle('selected', checkbox.checked);
                    // 이벤트 버블링 중단하여 체크박스 자체 클릭 방지 (이미 label이 처리하므로)
                    event.preventDefault();
                });
            }
        });
    }

    // --- 옵션 선택 페이지 (script_option.html) - 선택 시 스타일 변경 ---
    const optionItems = document.querySelectorAll('.option-item');
    if (optionItems.length > 0) {
        optionItems.forEach(item => {
            const radio = item.querySelector('input[type="radio"]');
            if (radio) {
                // 페이지 로드 시 라디오 버튼 상태에 따라 'selected' 클래스 적용
                if (radio.checked) {
                    item.classList.add('selected');
                }

                item.addEventListener('click', (event) => {
                    // 동일 그룹 내 모든 'selected' 클래스 제거
                    document.querySelectorAll(`input[name="${radio.name}"]`).forEach(otherRadio => {
                        otherRadio.closest('.option-item').classList.remove('selected');
                    });

                    // 클릭된 라디오 버튼의 'selected' 클래스 추가
                    radio.checked = true; // 라디오 버튼 선택
                    item.classList.add('selected');
                    event.preventDefault(); // 기본 라디오 버튼 클릭 동작 방지
                });
            }
        });
    }
});
