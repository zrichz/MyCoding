import feedparser

#news feeds:

'''
Reuters Top News: http://feeds.reuters.com/reuters/topNews
CNN Top Stories: http://rss.cnn.com/rss/edition.rss
The Guardian World News: https://www.theguardian.com/world/rss
Al Jazeera English: http://www.aljazeera.com/xml/rss/all.xml
NPR News: https://www.npr.org/rss/rss.php?id=1001
Wired Top Stories: https://www.wired.com/feed/rss
Hacker News: https://news.ycombinator.com/rss
Engadget: https://www.engadget.com/rss.xml
Science Daily: https://www.sciencedaily.com/rss/all.xml
The Verge: https://www.theverge.com/rss/index.xml
BBC : http://feeds.bbci.co.uk/news/rss.xml
Private Eye: https://www.private-eye.co.uk/issue-1530/in-the-back
'''

# URL of the RSS feed you want to parse
rss_feed_url = 'https://github.com/explore/feed.xml'
# Parse the RSS feed
feed = feedparser.parse(rss_feed_url)

# Print the feed title
print('Feed Title:', feed.feed.title)

# Iterate over the entries (articles) in the feed
for entry in feed.entries:
    print(entry.title)
    #print('Link:', entry.link)
    print(entry.published)
    print(entry.summary)
    #print('Author:', entry.author)
    print('Content:', entry.content[0].value if 'content' in entry else 'No content available')
    #print('ID:', entry.id)
    #print('Updated:', entry.updated if 'updated' in entry else 'Not available')
    #print('Tags:', [tag.term for tag in entry.tags] if 'tags' in entry else 'No tags available')
    print('-' * 100)

