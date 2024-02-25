chrome.runtime.onInstalled.addListener(function() {
    chrome.action.onClicked.addListener(async (tab) => {
      const tabs = await chrome.tabs.query({ currentWindow: true });
  
      for (const tab of tabs) {
        const text = await fetchPageContent(tab.id);
        if (text) {
          const result = await fetchHateSpeechDetectionAPI(text);
          console.log(result);
        }
      }
    });
  });
  
  async function fetchPageContent(tabId) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.scripting.executeScript({
      target: { tabId },
      function: () => document.body.innerText,
    });
    return response[0].result;
  }
  
  async function fetchHateSpeechDetectionAPI(text) {
    const response = await fetch('https://yaha-apna-api-daalo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    return data;
  }
  