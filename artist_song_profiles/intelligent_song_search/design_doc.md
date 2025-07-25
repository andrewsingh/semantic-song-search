Ok, so now I need a bit of help from you in designing this song search app. The central focus of the page, and the primary element that the user will interact with, should be a search bar. Then below the search bar should be the query song (for song-to-song search only - for text-to-song search the query will stay in the search bar). Then below that should be a scrollable list of song results. And finally, fixed to the bottom of the page view should be a Spotify web player to allow playing music directly in the app. Here are some more details about each component:

## Search Bar

- Beside the search bar (maybe one to the right, the other to the left) should be two elements for configuration:
    - An element to configure the type of search (song-based or text-based)
    - An element to configure what type of song embeddings to use for retrieval (5 options: full profile, sound aspect, meaning aspect, mood aspect, tags + genres). While some of these embeddings are more suited to one type of search over the other, I’d like to still have the option to use any of these 5 embedding types for both kinds of search, just to test it out.
    - This means there should be a total of 10 configurations for search: 2 types of search * 5 types of embeddings.
    - I’m thinking the elements should probably be dropdowns - seems like this would be the most straightforward.

### Text-to-song search

- For text-to-song search, the query can be freeform text - no restrictions. Once the user hits enter or submit, the current query (if there is a query) should be sent to OpenAI’s "text-embedding-3-large” to get embedded. Once the embedding is received, it should be used for kNN retrieval on the vector index that corresponds to the current embedding type.

### Song-to-song search

- For song-to-song search, the query must be an existing song in the library. So when the user enters a text-based query in the search bar, a dropdown list of query song suggestions should appear below the search bar, sorted in order of closest match to the query. Let’s call the function that fetches the top matching song results based on a text query the “matching function”.
- The matching function should support search by song title, by artist, and by album.
- The matching function should support “fuzzy matching”. That is, the query string doesn’t need to exactly match the song, artist, or album name - it only needs to be close.

### kNN retrieval

- Both types of search involve kNN retrieval from an embedding index. Note that we have 5 different types of embeddings that may be used in this search app. I’ll leave the implementation details of building this index (or multiple indices) to you. Try to implement this in a clean and modular way, so that I can easily add new types of embeddings in the future, and in a way that scales well as we expand the song library (right now it’s only 1500 songs).

## Query song card

- This should only be included for song-to-song search. Basically, it should be a song card in the same style as the song cards displayed in the search results, but shown in a special place to indicate that it is the query song. More details and features of the song cards will be explained in the results section below. Just know that the query song card should match the style and functionality of the result song cards, and should only exist when doing song-based search.

## Song results

- This should simply be a scrollable list of song result cards. Each result card is a single song, and should include the following: cover art, song title, artist title, album title, similarity score.

## Web player integration

- Each song that appears on the search app should also be playable. We will implement this by integrating a web player within the search app using the Spotify Web Playback SDK.
- Each song result card should be clickable such that by clicking on it, that song now starts playing on the web player.
- The web player should sit at the bottom of the app and be fixed in position. It should display the cover art and metadata of the currently playing song, as well as a progress bar to allow the user to select different parts of the song to play.

### Auth flow

- As part of the web player integration, the user will need to authenticate into their Spotify premium account. You should implement this auth flow, similar to how it’s implemented in the reference song2song_search_compare.py app.

## Design and styling

- The overall theme of the app should be a dark theme (dark mode).
- Song result cards should be hoverable, and change background color on hover.
- Only the song results should be scrollable: the search bar at the top and the web player at the bottom should be fixed.

## Important data filepaths

- Song profiles: pop_eval_set_v0/pop_eval_set_v0_results_enriched.json
    - I’ve enriched the song profiles with metadata from Spotify. The metadata for each song can be accessed by the “metadata” field. Use the values from this metadata object to populate the song result cards. I’ve also included a couple examples of song profiles so you can see the schema.
- Song profile embeddings: pop_eval_set_v0/pop_eval_set_v0_embeddings.npz.
    - These are saved in the same organizational scheme that was implemented in the embed_song_profiles.py script. Feel free to reference that script to understand how to properly load and match the embeddings with their original text.

Many of these requirements for this intelligent song search app, especially those for the web player integration, are very similar to that of my previous song2song_search_compare.py app that I had built previously. Feel free to reference the code for that app - including the Python script as well as any of the frontend code in the static/ directory - as you implement this app. Be sure to note the differences between this app and that one though, and feel free to improve on any part of that implementation if you think there's a better way to do something.

You’ll likely need to create multiple new files for this implementation: one Python file for the main app, and a few frontend files in a static/ subdirectory. I’ve created a new subdirectory within `artist_song_profiles` called `intelligent_song_search`. The full path to this directory is `/Users/andrew/dev/apollo/artist_song_profiles/intelligent_song_search` Feel free to put all the new files for this song search app in here. 

Does this all make sense? Let me know if you have any additional questions or need any clarification as you go through and implement the MVP for this intelligent song search app. Excited to see what you’ll spin up!