import React, { useEffect } from 'react';
import { GISCUS, SITE } from '../config';

const Comments = () => {
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.setAttribute('data-repo', GISCUS.repo);
    script.setAttribute('data-repo-id', GISCUS.repoId);
    script.setAttribute('data-category', GISCUS.category);
    script.setAttribute('data-category-id', GISCUS.categoryId);
    script.setAttribute('data-mapping', GISCUS.mapping || 'pathname');
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', GISCUS.reactionsEnabled?.toString() || '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', GISCUS.inputPosition || 'bottom');
    script.setAttribute('data-theme', SITE.lightAndDarkMode ? 'light' : 'dark');
    script.setAttribute('data-lang', 'ko'); // Since site lang is ko

    const commentsDiv = document.getElementById('comments-container');
    if (commentsDiv) {
      commentsDiv.appendChild(script);

      // Update theme on change if needed
      const updateTheme = (theme: string) => {
        const iframe = document.querySelector('iframe.giscus-frame') as HTMLIFrameElement;
        if (iframe) {
          iframe.contentWindow?.postMessage({ giscus: { setConfig: { theme } } }, 'https://giscus.app');
        }
      };

      const observer = new MutationObserver(() => {
        const currentTheme = SITE.lightAndDarkMode ? 'light' : 'dark';
        updateTheme(currentTheme);
      });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    }

    return () => {
      const commentsDiv = document.getElementById('comments-container');
      if (commentsDiv) {
        commentsDiv.innerHTML = '';
      }
    };
  }, []);

  return <div id="comments-container" />;
};

export default Comments;