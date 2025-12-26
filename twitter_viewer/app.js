/**
 * SemEval Twitter Thread Viewer
 * Interactive viewer for rumor stance detection dataset
 */

// State
let data = null;
let currentTopic = null;

// DOM Elements
const topicNav = document.getElementById('topic-nav');
const topicTitle = document.getElementById('topic-title');
const threadCount = document.getElementById('thread-count');
const tweetsContainer = document.getElementById('tweets-container');

// Topic icons for visual distinction
const topicIcons = {
    'charliehebdo': '🇫🇷',
    'ebola-essien': '🦠',
    'ferguson': '⚖️',
    'germanwings-crash': '✈️',
    'ottawashooting': '🍁',
    'prince-toronto': '👑',
    'putinmissing': '🇷🇺',
    'sydneysiege': '🇦🇺',
    'test': '🧪'
};

/**
 * Initialize the application
 */
async function init() {
    try {
        const response = await fetch('data.json');
        if (!response.ok) {
            throw new Error('Failed to load data');
        }
        data = await response.json();

        renderTopicNav();

        // Select first topic
        const firstTopic = Object.keys(data.topics)[0];
        if (firstTopic) {
            selectTopic(firstTopic);
        }
    } catch (error) {
        console.error('Error loading data:', error);
        tweetsContainer.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h2v2h-2v-2zm0-8h2v6h-2V9z"/>
                </svg>
                <p>Failed to load tweet data.</p>
                <p style="margin-top: 8px; font-size: 0.9rem;">Make sure to run <code>python prepare_data.py</code> first.</p>
            </div>
        `;
    }
}

/**
 * Render the topic navigation sidebar
 */
function renderTopicNav() {
    topicNav.innerHTML = '';

    Object.entries(data.topics).forEach(([key, topic]) => {
        const btn = document.createElement('button');
        btn.className = 'topic-btn';
        btn.dataset.topic = key;
        btn.innerHTML = `
            <span class="topic-icon">${topicIcons[key] || '📌'}</span>
            <span>${topic.name}</span>
            <span class="count">${topic.thread_count}</span>
        `;
        btn.addEventListener('click', () => selectTopic(key));
        topicNav.appendChild(btn);
    });
}

/**
 * Count all tweets in a thread recursively
 */
function countTweetsInThread(tweet) {
    let count = 1; // Count this tweet
    if (tweet.replies && tweet.replies.length > 0) {
        tweet.replies.forEach(reply => {
            count += countTweetsInThread(reply);
        });
    }
    return count;
}

/**
 * Count stance occurrences in a thread recursively
 */
function countStancesInThread(tweet, counts) {
    const stance = tweet.stance ? tweet.stance.toLowerCase() : 'comment';
    if (stance === 'support') counts.support++;
    else if (stance === 'deny') counts.deny++;
    else if (stance === 'query') counts.query++;
    else if (stance === 'source') counts.source++;
    else counts.comment++;

    if (tweet.replies && tweet.replies.length > 0) {
        tweet.replies.forEach(reply => {
            countStancesInThread(reply, counts);
        });
    }
    return counts;
}

/**
 * Get statistics for a topic
 */
function getTopicStats(threads) {
    let totalTweets = 0;
    const stanceCounts = { support: 0, deny: 0, query: 0, comment: 0, source: 0 };

    threads.forEach(thread => {
        totalTweets += countTweetsInThread(thread);
        countStancesInThread(thread, stanceCounts);
    });

    return { totalTweets, stanceCounts };
}

/**
 * Update legend with stance counts and percentages
 */
function updateLegend(stanceCounts, totalTweets) {
    const legendItems = document.querySelectorAll('.legend-item');
    legendItems.forEach(item => {
        const badge = item.querySelector('.stance-badge');
        if (!badge) return;

        let count = 0;
        let countSpan = item.querySelector('.stance-count');

        if (badge.classList.contains('support')) count = stanceCounts.support;
        else if (badge.classList.contains('deny')) count = stanceCounts.deny;
        else if (badge.classList.contains('query')) count = stanceCounts.query;
        else if (badge.classList.contains('comment')) count = stanceCounts.comment;

        if (!countSpan) {
            countSpan = document.createElement('span');
            countSpan.className = 'stance-count';
            item.appendChild(countSpan);
        }

        const percentage = totalTweets > 0 ? Math.round((count / totalTweets) * 100) : 0;
        countSpan.textContent = `${count} (${percentage}%)`;
    });
}

/**
 * Select and display a topic
 */
function selectTopic(topicKey) {
    currentTopic = topicKey;

    // Update active state
    document.querySelectorAll('.topic-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.topic === topicKey);
    });

    const topic = data.topics[topicKey];
    topicTitle.textContent = topic.name;

    // Calculate stats
    const stats = getTopicStats(topic.threads || []);
    threadCount.textContent = `${topic.thread_count} threads • ${stats.totalTweets} tweets`;

    // Update legend with stance counts
    updateLegend(stats.stanceCounts, stats.totalTweets);

    renderThreads(topic.threads);
}

/**
 * Render all threads for a topic
 */
function renderThreads(threads) {
    if (!threads || threads.length === 0) {
        tweetsContainer.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
                <p>No threads in this topic</p>
            </div>
        `;
        return;
    }

    tweetsContainer.innerHTML = '';

    threads.forEach(thread => {
        const threadElement = renderThread(thread);
        tweetsContainer.appendChild(threadElement);
    });
}

/**
 * Render context section (Wikipedia and URLs) as collapsible dropdowns
 */
function renderContext(context) {
    if (!context) return '';

    let html = '<div class="context-section">';

    // Wikipedia context - collapsible
    if (context.wikipedia) {
        html += `
            <button class="context-toggle wikipedia-toggle" onclick="toggleContext(this)">
                <img src="https://upload.wikimedia.org/wikipedia/commons/6/61/Wikipedia-logo-transparent.png" class="context-icon wikipedia-logo" alt="Wikipedia">
                <span>Wikipedia Context</span>
                <svg class="toggle-arrow" viewBox="0 0 24 24">
                    <path d="M12 15.121l-4.879-4.879-1.414 1.414L12 18.007l6.293-6.351-1.414-1.414z"/>
                </svg>
            </button>
            <div class="context-content" style="display: none;">
                <p class="context-text">${escapeHtml(context.wikipedia)}${context.wikipedia.length >= 495 ? '...' : ''}</p>
            </div>
        `;
    }

    // URLs context - collapsible
    if (context.urls && context.urls.length > 0) {
        html += `
            <button class="context-toggle urls-toggle" onclick="toggleContext(this)">
                <svg viewBox="0 0 24 24" class="context-icon">
                    <path d="M11.96 14.945c-.067 0-.136-.01-.203-.027-1.13-.318-2.097-.986-2.795-1.932-.832-1.125-1.176-2.508-.968-3.893s.942-2.605 2.068-3.438l3.53-2.608c2.322-1.716 5.61-1.224 7.33 1.1.83 1.127 1.175 2.51.967 3.895s-.943 2.605-2.07 3.438l-1.48 1.094c-.333.246-.804.175-1.05-.158-.246-.334-.176-.804.158-1.05l1.48-1.095c.803-.592 1.327-1.463 1.476-2.45.148-.988-.098-1.975-.69-2.778-1.225-1.656-3.572-2.01-5.23-.784l-3.53 2.608c-.802.593-1.326 1.464-1.475 2.45-.15.99.097 1.975.69 2.778.498.675 1.187 1.15 1.992 1.377.4.114.633.528.52.928-.092.33-.394.547-.722.547zm7.44-3.94c-.082 0-.165-.018-.242-.054-.35-.164-.504-.583-.34-.933.147-.313.232-.65.252-.994.062-1.045-.362-2.054-1.16-2.762-.253-.224-.282-.61-.059-.863.224-.252.61-.282.863-.058 1.105.977 1.69 2.374 1.603 3.821-.027.475-.144.94-.35 1.382-.118.256-.373.41-.642.41zm-5.74 9.95c-1.126-.317-2.096-.986-2.794-1.932-.832-1.125-1.176-2.508-.968-3.893.209-1.386.943-2.605 2.07-3.438l1.478-1.094c.333-.246.804-.175 1.05.158.246.334.175.804-.158 1.05l-1.48 1.095c-.803.593-1.327 1.464-1.476 2.45-.15.99.098 1.976.69 2.778.5.675 1.187 1.15 1.993 1.377.4.114.633.528.52.928-.092.33-.394.547-.722.547-.07 0-.14-.01-.208-.027z"/>
                </svg>
                <span>Referenced URLs (${context.urls.length})</span>
                <svg class="toggle-arrow" viewBox="0 0 24 24">
                    <path d="M12 15.121l-4.879-4.879-1.414 1.414L12 18.007l6.293-6.351-1.414-1.414z"/>
                </svg>
            </button>
            <div class="context-content" style="display: none;">
                <div class="context-urls">
                    ${context.urls.map(url => `
                        <a href="${escapeHtml(url.expanded_url)}" target="_blank" rel="noopener noreferrer" class="context-url">
                            ${escapeHtml(url.expanded_url.length > 80 ? url.expanded_url.substring(0, 80) + '...' : url.expanded_url)}
                        </a>
                    `).join('')}
                </div>
            </div>
        `;
    }

    html += '</div>';
    return html;
}

/**
 * Toggle context dropdown visibility
 */
function toggleContext(btn) {
    const content = btn.nextElementSibling;
    const isExpanded = content.style.display !== 'none';
    content.style.display = isExpanded ? 'none' : 'block';
    btn.classList.toggle('expanded', !isExpanded);
}

/**
 * Render a single thread with its source tweet and replies
 */
function renderThread(tweet, isReply = false) {
    const div = document.createElement('div');
    div.className = `tweet-card ${isReply ? 'reply-tweet' : 'source-tweet'}`;

    const hasReplies = tweet.replies && tweet.replies.length > 0;
    const stanceClass = getStanceClass(tweet.stance);
    const stanceLetter = getStanceLetter(tweet.stance);

    div.innerHTML = `
        <div class="tweet-header">
            <div class="avatar">
                ${getAvatarContent(tweet.user)}
            </div>
            <div class="tweet-user-info">
                <div class="tweet-user-row">
                    <span class="user-name">${escapeHtml(tweet.user.name)}</span>
                    ${tweet.user.verified ? `
                        <svg class="verified-badge" viewBox="0 0 24 24">
                            <path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.998-3.818-3.998-.47 0-.92.084-1.336.25C14.818 2.415 13.51 1.5 12 1.5s-2.816.917-3.437 2.25c-.415-.165-.866-.25-1.336-.25-2.11 0-3.818 1.79-3.818 4 0 .494.083.964.237 1.4-1.272.65-2.147 2.018-2.147 3.6 0 1.495.782 2.798 1.942 3.486-.02.17-.032.34-.032.514 0 2.21 1.708 4 3.818 4 .47 0 .92-.086 1.335-.25.62 1.334 1.926 2.25 3.437 2.25 1.512 0 2.818-.916 3.437-2.25.415.163.865.248 1.336.248 2.11 0 3.818-1.79 3.818-4 0-.174-.012-.344-.033-.513 1.158-.687 1.943-1.99 1.943-3.484zm-6.616-3.334l-4.334 6.5c-.145.217-.382.334-.625.334-.143 0-.288-.04-.416-.126l-.115-.094-2.415-2.415c-.293-.293-.293-.768 0-1.06s.768-.294 1.06 0l1.77 1.767 3.825-5.74c.23-.345.696-.436 1.04-.207.346.23.44.696.21 1.04z"/>
                        </svg>
                    ` : ''}
                    <span class="user-handle">@${escapeHtml(tweet.user.screen_name)}</span>
                    <span class="tweet-date">${tweet.created_at}</span>
                </div>
            </div>
            <span class="stance-badge ${stanceClass}" title="${tweet.stance}">${stanceLetter}</span>
        </div>
        <div class="tweet-content">
            <p class="tweet-text">${formatTweetText(tweet.text)}</p>
        </div>
        <div class="tweet-meta">
            <span class="tweet-stat replies" title="Replies">
                <svg viewBox="0 0 24 24">
                    <path d="M1.751 10c0-4.42 3.584-8 8.005-8h4.366c4.49 0 8.129 3.64 8.129 8.13 0 2.96-1.607 5.68-4.196 7.11l-8.054 4.46v-3.69h-.067c-4.49.1-8.183-3.51-8.183-8.01zm8.005-6c-3.317 0-6.005 2.69-6.005 6 0 3.37 2.77 6.08 6.138 6.01l.351-.01h1.761v2.3l5.087-2.81c1.951-1.08 3.163-3.13 3.163-5.36 0-3.39-2.744-6.13-6.129-6.13H9.756z"/>
                </svg>
                ${hasReplies ? tweet.replies.length : 0}
            </span>
            <span class="tweet-stat retweets" title="Retweets">
                <svg viewBox="0 0 24 24">
                    <path d="M4.5 3.88l4.432 4.14-1.364 1.46L5.5 7.55V16c0 1.1.896 2 2 2H13v2H7.5c-2.209 0-4-1.79-4-4V7.55L1.432 9.48.068 8.02 4.5 3.88zM16.5 6H11V4h5.5c2.209 0 4 1.79 4 4v8.45l2.068-1.93 1.364 1.46-4.432 4.14-4.432-4.14 1.364-1.46 2.068 1.93V8c0-1.1-.896-2-2-2z"/>
                </svg>
                ${tweet.retweet_count || 0}
            </span>
            <span class="tweet-stat likes" title="Likes">
                <svg viewBox="0 0 24 24">
                    <path d="M16.697 5.5c-1.222-.06-2.679.51-3.89 2.16l-.805 1.09-.806-1.09C9.984 6.01 8.526 5.44 7.304 5.5c-1.243.07-2.349.78-2.91 1.91-.552 1.12-.633 2.78.479 4.82 1.074 1.97 3.257 4.27 7.129 6.61 3.87-2.34 6.052-4.64 7.126-6.61 1.111-2.04 1.03-3.7.477-4.82-.561-1.13-1.666-1.84-2.908-1.91zm4.187 7.69c-1.351 2.48-4.001 5.12-8.379 7.67l-.503.3-.504-.3c-4.379-2.55-7.029-5.19-8.382-7.67-1.36-2.5-1.41-4.86-.514-6.67.887-1.79 2.647-2.91 4.601-3.01 1.651-.09 3.368.56 4.798 2.01 1.429-1.45 3.146-2.1 4.796-2.01 1.954.1 3.714 1.22 4.601 3.01.896 1.81.846 4.17-.514 6.67z"/>
                </svg>
                ${tweet.favorite_count || 0}
            </span>
            <span class="tweet-meta-right">
                <span class="tweet-id" title="Tweet ID">${tweet.id}</span>
                <button class="info-btn" title="View metadata" data-metadata='${escapeHtml(JSON.stringify(getMetadata(tweet)))}'>
                    <svg viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                    </svg>
                </button>
            </span>
        </div>
        ${!isReply && tweet.context ? renderContext(tweet.context) : ''}
    `;

    // Add replies if present
    if (hasReplies) {
        const repliesContainer = document.createElement('div');
        repliesContainer.className = 'reply-thread';
        repliesContainer.style.display = 'none';

        tweet.replies.forEach(reply => {
            const replyElement = renderThread(reply, true);
            repliesContainer.appendChild(replyElement);
        });

        // Add toggle button inside the tweet card (before the divider)
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'thread-toggle';
        toggleBtn.innerHTML = `
            <svg viewBox="0 0 24 24">
                <path d="M12 15.121l-4.879-4.879-1.414 1.414L12 18.007l6.293-6.351-1.414-1.414z"/>
            </svg>
            <span>Show ${tweet.replies.length} ${tweet.replies.length === 1 ? 'reply' : 'replies'}</span>
        `;

        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isExpanded = repliesContainer.style.display !== 'none';
            repliesContainer.style.display = isExpanded ? 'none' : 'block';
            toggleBtn.classList.toggle('expanded', !isExpanded);
            toggleBtn.querySelector('span').textContent = isExpanded
                ? `Show ${tweet.replies.length} ${tweet.replies.length === 1 ? 'reply' : 'replies'}`
                : `Hide ${tweet.replies.length} ${tweet.replies.length === 1 ? 'reply' : 'replies'}`;
        });

        // Append toggle button inside the tweet card div
        div.appendChild(toggleBtn);

        const wrapper = document.createElement('div');
        wrapper.className = 'tweet-thread-wrapper';
        wrapper.appendChild(div);
        wrapper.appendChild(repliesContainer);

        return wrapper;
    }

    return div;
}

/**
 * Get stance CSS class
 */
function getStanceClass(stance) {
    if (!stance) return 'comment';
    const s = stance.toLowerCase();
    if (s === 'support') return 'support';
    if (s === 'deny') return 'deny';
    if (s === 'query') return 'query';
    return 'comment';
}

/**
 * Get stance letter abbreviation
 */
function getStanceLetter(stance) {
    if (!stance) return 'C';
    const s = stance.toLowerCase();
    if (s === 'support') return 'S';
    if (s === 'deny') return 'D';
    if (s === 'query') return 'Q';
    if (s === 'source') return '★';
    return 'C';
}

/**
 * Get avatar content (image or initials)
 */
function getAvatarContent(user) {
    if (user.profile_image_url && !user.profile_image_url.includes('default_profile')) {
        return `<img src="${escapeHtml(user.profile_image_url)}" alt="${escapeHtml(user.name)}" onerror="this.style.display='none'; this.parentElement.textContent='${getInitials(user.name)}'">`;
    }
    return getInitials(user.name);
}

/**
 * Get initials from name
 */
function getInitials(name) {
    if (!name) return '?';
    return name.split(' ')
        .map(word => word[0])
        .slice(0, 2)
        .join('')
        .toUpperCase();
}

/**
 * Format tweet text with links and mentions
 */
function formatTweetText(text) {
    if (!text) return '';

    let formatted = escapeHtml(text);

    // Convert URLs to links
    formatted = formatted.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Convert @mentions to links
    formatted = formatted.replace(
        /@(\w+)/g,
        '<a href="https://twitter.com/$1" target="_blank" rel="noopener noreferrer">@$1</a>'
    );

    // Convert #hashtags to links
    formatted = formatted.replace(
        /#(\w+)/g,
        '<a href="https://twitter.com/hashtag/$1" target="_blank" rel="noopener noreferrer">#$1</a>'
    );

    return formatted;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Get metadata object for a tweet (fields not otherwise displayed)
 */
function getMetadata(tweet) {
    return {
        'Tweet ID': tweet.id,
        'In Reply To': tweet.in_reply_to_status_id || 'N/A',
        'Stance': tweet.stance || 'unknown',
        'User Handle': '@' + tweet.user.screen_name,
        'Followers': tweet.user.followers_count?.toLocaleString() || '0',
        'Verified': tweet.user.verified ? 'Yes' : 'No'
    };
}

/**
 * Show metadata popup
 */
function showMetadata(btn, metadataJson) {
    // Remove any existing popups
    const existing = document.querySelector('.metadata-popup');
    if (existing) existing.remove();

    const metadata = JSON.parse(metadataJson);

    const popup = document.createElement('div');
    popup.className = 'metadata-popup';

    let content = '<h4>Tweet Metadata</h4><div class="metadata-list">';
    for (const [key, value] of Object.entries(metadata)) {
        content += `<div class="metadata-row"><span class="metadata-key">${key}:</span><span class="metadata-value">${value}</span></div>`;
    }
    content += '</div><button class="metadata-close" onclick="this.parentElement.remove()">×</button>';

    popup.innerHTML = content;
    btn.parentElement.appendChild(popup);

    // Close when clicking outside
    setTimeout(() => {
        document.addEventListener('click', function closePopup(e) {
            if (!popup.contains(e.target) && e.target !== btn) {
                popup.remove();
                document.removeEventListener('click', closePopup);
            }
        });
    }, 10);
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);

// Event delegation for info buttons
document.addEventListener('click', (e) => {
    const infoBtn = e.target.closest('.info-btn');
    if (infoBtn) {
        e.stopPropagation();
        const metadataStr = infoBtn.dataset.metadata;
        if (metadataStr) {
            showMetadata(infoBtn, metadataStr);
        }
    }
});
