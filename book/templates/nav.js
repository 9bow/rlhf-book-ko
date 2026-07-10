class NavigationDropdown extends HTMLElement {
    constructor() {
      super();

      // Get the initial expanded state from the attribute, default to false
      const initialExpanded = this.getAttribute('expanded') === 'true';
      const contentId = `navigation-dropdown-content-${NavigationDropdown.nextId++}`;

      this.innerHTML = `
        <div>
          <button class="dropdown-button" aria-expanded="${initialExpanded}" aria-controls="${contentId}">
            <span><strong>Navigation</strong></span>
            <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>

          <div id="${contentId}" class="dropdown-content${initialExpanded ? ' open' : ''}">
    <nav class="chapter-nav">
      <div class="section">
        <h3>Links (한국어)</h3>
        <ul>
          <li><a href="https://9bow.github.io/rlhf-book-ko">홈</a> / <a href="https://github.com/9bow/rlhf-book-ko">GitHub</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/book.pdf">PDF</a> / <a href="https://9bow.github.io/rlhf-book-ko/book.epub">EPUB</a> / <a href="https://9bow.github.io/rlhf-book-ko/book.kindle.epub">Kindle</a></li>
        </ul>
        <h3>Links</h3>
        <ul>
          <li><a href="https://9bow.github.io/rlhf-book-ko">Home</a> / <a href="https://github.com/natolambert/rlhf-book">GitHub</a> / <a href="https://discord.gg/yz5AwK4gBR">Discord</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501">arXiv</a> / <a href="https://9bow.github.io/rlhf-book-ko/book.epub">EPUB</a> / <a href="https://9bow.github.io/rlhf-book-ko/book.kindle.epub">Kindle</a></li>
          <li>Order: <a href="https://hubs.la/Q03TsMBq0">Manning</a>, <a href="https://amzn.to/4cwCDJQ">Amazon</a></li>
        </ul>
        <h3>자료</h3>
        <ul>
          <li><a href="https://9bow.github.io/rlhf-book-ko/rl-cheatsheet">RL 치트시트</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/library">모델 완성 비교</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/course">연계 강좌</a></li>
        </ul>
      </div>

      <div class="section">
        <h3>도입부</h3>
        <ol start="1">
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/01-introduction">소개</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/02-related-works">RLHF의 짧은 역사</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/03-training-overview">학습 개요</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>핵심 학습 파이프라인</h3>
        <ol start="4">
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/04-instruction-tuning">지시 미세조정</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/instruction_tuning">code</a>]</li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/05-reward-models">보상 모델링</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/reward_models">code</a>]</li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/06-policy-gradients">강화학습</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/policy_gradients">code</a>]</li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/07-reasoning">추론과 추론 시간 스케일링</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/08-direct-alignment">직접 정렬 알고리즘</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/direct_alignment">code</a>]</li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/09-rejection-sampling">거부 샘플링</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/rejection_sampling">code</a>]</li>
        </ol>
      </div>

      <div class="section">
        <h3>데이터와 선호도</h3>
        <ol start="10">
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/10-preferences">선호도의 본질</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/11-preference-data">선호도 데이터</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/12-synthetic-data">합성 데이터와 증류</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/distillation">code</a>]</li>
        </ol>
      </div>

      <div class="section">
        <h3>실무 고려사항</h3>
        <ol start="13">
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/13-tools">도구 사용 및 함수 호출</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/14-over-optimization">과최적화</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/15-regularization">정규화</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/16-evaluation">평가</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/17-product">모델 캐릭터와 제품</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>부록</h3>
        <ol type="A" style="padding-left: 0; list-style-position: inside;">
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/appendix-a-definitions">정의</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/appendix-b-style">"단순한 스타일"을 넘어서</a></li>
          <li><a href="https://9bow.github.io/rlhf-book-ko/c/appendix-c-practical">실무 이슈</a></li>
        </ol>
      </div>
    </nav>
    <div id="search"></div>
  </div>
</div>
      `;

      // Initialize Pagefind search if available
      var searchEl = this.querySelector('#search');
      if (searchEl && typeof PagefindUI !== 'undefined') {
        new PagefindUI({ element: searchEl, showImages: false });
      }

      this.markCurrentPage();

      // Set up click handler
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');

      button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', !isExpanded);
        content.classList.toggle('open');
      });
    }

    // Add attribute change observer
    static get observedAttributes() {
      return ['expanded'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (name === 'expanded') {
        const button = this.querySelector('.dropdown-button');
        const content = this.querySelector('.dropdown-content');
        const isExpanded = newValue === 'true';

        if (button && content) {
          button.setAttribute('aria-expanded', isExpanded);
          content.classList.toggle('open', isExpanded);
        }
      }
    }

    markCurrentPage() {
      const normalizePath = (path) => path.replace(/\/index\.html$/, '/').replace(/\.html$/, '').replace(/\/$/, '') || '/';
      const currentPath = normalizePath(window.location.pathname);

      this.querySelectorAll('a[href]').forEach((link) => {
        const url = new URL(link.getAttribute('href'), window.location.origin);
        if (url.origin !== window.location.origin) {
          return;
        }

        const linkPath = normalizePath(url.pathname);
        if (linkPath === currentPath) {
          link.setAttribute('aria-current', 'page');
        }
      });
    }
}

NavigationDropdown.nextId = 0;

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}
