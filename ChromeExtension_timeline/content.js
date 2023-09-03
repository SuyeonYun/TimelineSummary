"use strict";

// URL에서 검색 매개변수를 추출하는 함수
function getSearchParam(str) {

    const searchParam = (str && str !== "") ? str : window.location.search;

    if (!(/\?([a-zA-Z0-9_]+)/i.exec(searchParam))) return {};
    let match,
        pl     = /\+/g,  // 덧셈 기호를 공백으로 대체하기 위한 정규식
        search = /([^?&=]+)=?([^&]*)/g,
        decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
        index = /\?([a-zA-Z0-9_]+)/i.exec(searchParam)["index"]+1,
        videoId  = searchParam.substring(index);

    let urlParams = {};
    while (match = search.exec(videoId)) {
        urlParams[decode(match[1])] = decode(match[2]);
    }
    return urlParams;   
    
}

// 위젯 요소 초기화를 위한 함수
function sanitizeWidget() {

    // // 자막 관련 요소 초기화
    document.querySelector("#yt_timeline_select").innerHTML = "";
    document.querySelector("#yt_timeline_text").innerHTML = "";

    // 높이 조절
    document.querySelector("#yt_timeline_body").style.maxHeight = window.innerHeight - 160 + "px";
    document.querySelector("#yt_timeline_text").innerHTML = `
            <div class="yt_timeline_transcript_text_segment_beforeSelect">
                <div class="yt_timeline_transcript_beforeSelect">Select "script type"</div>
            </div>`

    // 클래스 리스트 토글
    document.querySelector("#yt_timeline_body").classList.toggle("yt_timeline_body_show");
    document.querySelector(".yt_timeline_header_actions").classList.toggle("yt_timeline_header_toggle_heart");
    
    // "기본이냥/자세히냥" 버튼 생성   
    document.querySelector("#yt_timeline_select").innerHTML = `
    <button class="yt_timeline_select_button" id = "yt_timeline_select_tight">ฅ기본이냥ฅ</button>
    <button class="yt_timeline_select_button" id = "yt_timeline_select_loose">ฅ자세히냥ฅ</button>`;
}

// 위젯이 열려 있는지 확인하는 함수
async function isWidgetOpen() {
    return document.querySelector("#yt_timeline_body").classList.contains("yt_timeline_body_show");
}

// "기본이냥" 버튼 클릭 이벤트 리스너 추가
function evtListenerOntightBtns(tight_script, videoId) {
    var Button1 = document.getElementById("yt_timeline_select_tight");
    var Button2 = document.getElementById("yt_timeline_select_loose");
    Button1.addEventListener("click", async () => {

        //버튼을 누르면 선택된 버튼으로써 활동 
        Button1.classList.add("yt_timeline_tight_selected");
        if (document.querySelector('.yt_timeline_loose_selected')) {Button2.classList.remove("yt_timeline_loose_selected");}

        //서버로부터 받은 파일이 없으면 로딩화면을 띄운다.
        if (!tight_script) { 
            document.querySelector("#yt_timeline_text").innerHTML = `
            <svg class="yt_timeline_loading" style="display: block;width: 48px;margin: 40px auto;" width="48" height="48" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M100 36C59.9995 36 37 66 37 99C37 132 61.9995 163.5 100 163.5C138 163.5 164 132 164 99" stroke="#5C94FF" stroke-width="6"/>
            </svg>`;

            //"기본이냥" script가 참값이 될 때까지 대기
            await waitForTruthyValue(tight_script, 100) // Check every 100 milliseconds
        }
        
        //버튼을 눌렀을 때 파일을 서버로부터 받은 상황이면 "기본이냥" script 출력
        document.querySelector("#yt_timeline_text").innerHTML = ''
        const tight_HTML = getTranscriptHTML(tight_script, videoId)
        document.querySelector("#yt_timeline_text").innerHTML = tight_HTML
        evtListenerOnTimestamp()
    })
}

// "자세히냥" 버튼 클릭 이벤트 리스너 추가
function evtListenerOnlooseBtns(loose_script, videoId) {
    var Button1 = document.getElementById("yt_timeline_select_tight");
    var Button2 = document.getElementById("yt_timeline_select_loose");
    Button2.addEventListener("click", async () => {

        //버튼을 누르면 선택된 버튼으로써 활동
        Button2.classList.add("yt_timeline_loose_selected");
        if (document.querySelector('.yt_timeline_tight_selected')) {Button1.classList.remove("yt_timeline_tight_selected");}

        //서버로부터 받은 파일이 없으면 로딩화면을 띄운다.
        if (!loose_script) {
            document.querySelector("#yt_timeline_text").innerHTML = `
            <svg class="yt_timeline_loading" style="display: block;width: 48px;margin: 40px auto;" width="48" height="48" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M100 36C59.9995 36 37 66 37 99C37 132 61.9995 163.5 100 163.5C138 163.5 164 132 164 99" stroke="#5C94FF" stroke-width="6"/>
            </svg>`;

            await waitForTruthyValue(loose_script, 100) // Check every 100 milliseconds
        }
        
        //버튼을 눌렀을 때 파일을 서버로부터 받은 상황이면 "자세히냥" script 출력
        document.querySelector("#yt_timeline_text").innerHTML = ''
        const loose_HTML = await getTranscriptHTML(loose_script, videoId)
        document.querySelector("#yt_timeline_text").innerHTML = loose_HTML
        evtListenerOnTimestamp()
    })
}

// 자막 HTML 생성 함수
function getTranscriptHTML(inputText, videoId) {

    // 첫 글자가 숫자가 아니면 출력할 TimeLine이 없는 영상인 것 -> 없다고 출력할 것
    if (isNaN(parseInt(inputText[0]))) {
        return `
        <div class="yt_timeline_transcript_text_segment">
            <div class="yt_timeline_transcript_text">${inputText}</div>
        </div>`; 
    }

    // str 문장을 <br> 단위로 쪼개고 timestamp부분과 문장 부분으로 나누기
    const segments = inputText.split("<br>");
    const transcriptHTMLArray = segments.map(segment => {
        const parts = segment.trim().split(" ");
        const timestamp = parts[0];
        const text = parts.slice(1).join(" ").trim();
        const [mm, ss] = timestamp.trim().split(":", 2);
        const t = (Number(mm))*60 + Number(ss);
        
        // script를 새로 만든 형식으로 html text부분에 출력
        return `
            <div class="yt_timeline_transcript_text_segment">
                <div><a class="yt_timeline_transcript_text_timestamp" style="padding-top: 16px !important;" href="/watch?v=${videoId}&t=${t}s" target="_blank" data-timestamp-href="/watch?v=${videoId}&t=${t}s" data-start-time="${t}">${timestamp}</a></div>
                <div class="yt_timeline_transcript_text" data-start-time="${t}">${text}</div>
            </div>`;
    });

    return transcriptHTMLArray.join("");
}

// 타임스탬프 클릭 시 영상의 해당 지점으로 넘어갈 수 있도록 작동하는 함수
function evtListenerOnTimestamp() {
    Array.from(document.getElementsByClassName("yt_timeline_transcript_text_timestamp")).forEach(el => {
        el.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const starttime = el.getAttribute("data-start-time");
            const ytVideoEl = document.querySelector("#movie_player > div.html5-video-container > video");
            ytVideoEl.currentTime = starttime;
            ytVideoEl.play();
        })
    })
}

// 해당 페이지에 인자 요소가 나타날 때까지 대기 - 비동기식 접근에 용이
function waitForElm(selector) {
    return new Promise(resolve => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                resolve(document.querySelector(selector));
                observer.disconnect();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
}

// "variable"이 참값이 될 때까지 대기 - script를 받을 때 사용
function waitForTruthyValue(variable, interval) {
    return new Promise((resolve) => {
      const intervalId = setInterval(() => {
        if (variable) {
          clearInterval(intervalId);
          resolve(variable);
        }
      }, interval);
    });
}

// <MAIN FUNCTION> 버튼을 삽입하고, 시스템 내부 기능을 순서대로 실행 하는 함수
function insertBtn() {

    var loose_script;
    var tight_script;

    //videoId 추출
    var videoId = getSearchParam(window.location.href).v;

    // 자막 관련 요소를 초기화하고 제거
    if (document.querySelector("#yt_timeline_button")) { document.querySelector("#yt_timeline_button").innerHTML = ""; }
    if (document.querySelector("#yt_timeline_summary")) { document.querySelector("#yt_timeline_summary").innerHTML = ""; }
    Array.from(document.getElementsByClassName("yt_timeline_container")).forEach(el => { el.remove(); });

    // videoId가 없으면 영상이 없는 페이지이므로 함수 종료
    if (!getSearchParam(window.location.href).v) { return; }

    // YouTube 동영상 페이지의 요소가 로드될 때까지 대기
    waitForElm('#secondary.style-scope.ytd-watch-flexy').then(() => {

        // 초기화
        Array.from(document.getElementsByClassName("yt_timeline_container")).forEach(el => { el.remove(); });

        document.querySelector("#secondary.style-scope.ytd-watch-flexy").insertAdjacentHTML("afterbegin", `
        <head>
            <meta charset="UTF-8">  
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=Cute+Font&family=Do+Hyeon&family=East+Sea+Dokdo&family=Gothic+A1&family=Gugi&family=Nanum+Pen+Script&family=Noto+Serif+KR:wght@300&family=Poor+Story&family=Single+Day&family=Sunflower:wght@300&display=swap" rel="stylesheet">
        </head>
        <body>
            <div class="yt_timeline_container">
                <div id="yt_timeline_header" class="yt_timeline_header" data-hover-label="질문을 기준으로 타임라인을 생성합니다. 고양이를 눌러 시작하세요">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. -->
                        <path fill="#C92222" fill-opacity="0.9" d="M226.5 92.9c14.3 42.9-.3 86.2-32.6 96.8s-70.1-15.6-84.4-58.5s.3-86.2 32.6-96.8s70.1 15.6 84.4 58.5zM100.4 198.6c18.9 32.4 14.3 70.1-10.2 84.1s-59.7-.9-78.5-33.3S-2.7 179.3 21.8 165.3s59.7 .9 78.5 33.3zM69.2 401.2C121.6 259.9 214.7 224 256 224s134.4 35.9 186.8 177.2c3.6 9.7 5.2 20.1 5.2 30.5v1.6c0 25.8-20.9 46.7-46.7 46.7c-11.5 0-22.9-1.4-34-4.2l-88-22c-15.3-3.8-31.3-3.8-46.6 0l-88 22c-11.1 2.8-22.5 4.2-34 4.2C84.9 480 64 459.1 64 433.3v-1.6c0-10.4 1.6-20.8 5.2-30.5zM421.8 282.7c-24.5-14-29.1-51.7-10.2-84.1s54-47.3 78.5-33.3s29.1 51.7 10.2 84.1s-54 47.3-78.5 33.3zM310.1 189.7c-32.3-10.6-46.9-53.9-32.6-96.8s52.1-69.1 84.4-58.5s46.9 53.9 32.6 96.8s-52.1 69.1-84.4 58.5z"/>
                    </svg>   
                    <p class="yt_timeline_header_text">챕터를 알려줄게~얍!</p>
                    <div class="yt_timeline_header_actions">
                        <div style="filter: brightness(0.9);" id="yt_timeline_header_toggle" class="yt_timeline_header_action_btn">(=^ω^=)/''</div>
                    </div>
                </div>
                <div id="yt_timeline_body" class="yt_timeline_body">
                    <div id="yt_timeline_select" class="yt_timeline_select"></div>
                    <div id="yt_timeline_text" class="yt_timeline_text"></div>
                </div>
            </div>
        </body>`);

        // Youtube 페이지에 들어서자마자 서버와 통신하여 script 받아오기 -> "로딩하는 시간을 줄임"
        waitForElm('#yt_timeline_header').then(() => {
            function sendTightRequestUsingXHR(url) {
                var xhr = new XMLHttpRequest();
                xhr.open('GET', url, true);
        
                xhr.onload = function () {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        tight_script = xhr.responseText;
                        if (document.querySelector(".yt_timeline_tight_selected")){
                            document.querySelector("#yt_timeline_text").innerHTML = ''
                            const tight_HTML = getTranscriptHTML(tight_script, videoId)
                            document.querySelector("#yt_timeline_text").innerHTML = tight_HTML
                            evtListenerOnTimestamp()
                        }
                    } else {
                        console.error('Request failed with status:', xhr.status);
                    }
                };
        
                xhr.onerror = function () {
                    console.log('Request error occurred');
                };
        
                xhr.send();
            }
            //이미 "기본이냥" script가 존재할 경우 다시 로딩하는 일이 없도록 설계
            if (!tight_script) {
                sendTightRequestUsingXHR('http://127.0.0.1:5000/makemore?v=' + videoId)
            }
        });

        waitForElm('#yt_timeline_header').then(() => {

            function sendLooseRequestUsingXHR(url) {
                var xhr = new XMLHttpRequest();
                xhr.open('GET', url, true);
        
                xhr.onload = function () {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        loose_script = xhr.responseText;
                        if (!document.querySelector(".yt_timeline_loose_selected")){
                            document.querySelector("#yt_timeline_text").innerHTML = ''
                            const loose_HTML = getTranscriptHTML(loose_script, videoId)
                            document.querySelector("#yt_timeline_text").innerHTML = loose_HTML
                            evtListenerOnTimestamp()
                        }
                    } else {
                        console.error('Request failed with status:', xhr.status);
                    }
                };
        
                xhr.onerror = function () {
                    console.log('Request error occurred');
                };
        
                xhr.send();
            }
            //이미 "자세히냥" script가 존재할 경우 다시 로딩하는 일이 없도록 설계
            if (!loose_script) {
                sendLooseRequestUsingXHR('http://127.0.0.1:5000/makeTL?v=' + videoId)
            }
        });

        // 헤더에 덧붙일 설명을 hover을 활용하여 display
        var el = document.getElementById("yt_timeline_header")
        const label = el.getAttribute("data-hover-label");
        if (!label) { return; }
        el.addEventListener("mouseenter", (e) => {
            e.stopPropagation();
            e.preventDefault();
            Array.from(document.getElementsByClassName("yt_timeline_header_hover_label")).forEach(el => { el.remove(); })
            el.insertAdjacentHTML("beforeend", `<div class="yt_timeline_header_hover_label">${label.replace(/\n+/g, `<br />`)}</div>`);
        })
        el.addEventListener("mouseleave", (e) => {
            e.stopPropagation();
            e.preventDefault();
            Array.from(document.getElementsByClassName("yt_timeline_header_hover_label")).forEach(el => { el.remove(); })
        })

        document.querySelector(".yt_timeline_header_actions").addEventListener("click", async (e) => {
            
            sanitizeWidget();

            //toggle - 버튼 누를 때 고양이 그림 바꾸기
            if (document.querySelector('.yt_timeline_header_toggle_heart')){
                document.querySelector('.yt_timeline_header_actions').innerHTML = ''
                document.querySelector('.yt_timeline_header_actions').innerHTML = `
                <div style="filter: brightness(0.9);" id="yt_timeline_header_toggle" class="yt_timeline_header_action_btn">(<svg xmlns="http://www.w3.org/2000/svg" height="0.625em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path fill="#C92222" fill-opacity="0.9" d="M47.6 300.4L228.3 469.1c7.5 7 17.4 10.9 27.7 10.9s20.2-3.9 27.7-10.9L464.4 300.4c30.4-28.3 47.6-68 47.6-109.5v-5.8c0-69.9-50.5-129.5-119.4-141C347 36.5 300.6 51.4 268 84L256 96 244 84c-32.6-32.6-79-47.5-124.6-39.9C50.5 55.6 0 115.2 0 185.1v5.8c0 41.5 17.2 81.2 47.6 109.5z"/></svg>^ω^<svg xmlns="http://www.w3.org/2000/svg" height="0.625em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path fill="#C92222" fill-opacity="0.9" d="M47.6 300.4L228.3 469.1c7.5 7 17.4 10.9 27.7 10.9s20.2-3.9 27.7-10.9L464.4 300.4c30.4-28.3 47.6-68 47.6-109.5v-5.8c0-69.9-50.5-129.5-119.4-141C347 36.5 300.6 51.4 268 84L256 96 244 84c-32.6-32.6-79-47.5-124.6-39.9C50.5 55.6 0 115.2 0 185.1v5.8c0 41.5 17.2 81.2 47.6 109.5z"/></svg>)/''</div>`
            }
            else {
                document.querySelector('.yt_timeline_header_actions').innerHTML = `
                <div style="filter: brightness(0.9);" id="yt_timeline_header_toggle" class="yt_timeline_header_action_btn">(=^ω^=)/''</div>`
            }

            const widget = await isWidgetOpen();
            if (!widget) { return; }

            // 버튼 클릭시 이벤트를 실행할 수 있도록 설정
            evtListenerOntightBtns(tight_script, videoId);
            evtListenerOnlooseBtns(loose_script, videoId);
        });
    })
}

// main - Youtube 페이지 로드가 끝난 후, 시스템 실행
let oldHref = "";

window.onload = async () => {
        
    if (window.location.hostname === "www.youtube.com") {
        
        if (window.location.search !== "" && window.location.search.includes("v=")) {
            insertBtn();
        }

        const bodyList = document.querySelector("body");
        let observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (oldHref !== document.location.href) {
                    oldHref = document.location.href;
                    insertBtn();
                }
            });
        });
        observer.observe(bodyList, { childList: true, subtree: true });

    }
    
}

