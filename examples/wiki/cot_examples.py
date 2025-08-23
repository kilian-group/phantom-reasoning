"""Few-shot examples

From Memento (add link)
"""

# ------------------------------------------------------------------------------------------------
# HotpotQA
# ------------------------------------------------------------------------------------------------
COT_EXAMPLES_HP = f"""\
Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: First I need to find the year Arthur's Magazine was started. Based on the evidence, Arthur's Magazine was started in 1844. Next, I need to find the year First for Women was started. Based on the evidence, First for Women was started in 1989. Since 1844 is before 1989, Arthur's Magazine was started first. <answer>Arthur's Magazine</answer>.

Question: The Oberoi family is part of a hotel company that has a head office in what city?
Answer: First I need to find what hotel company the Oberoi family is part of. Based on the evidence, the Oberoi family is part of the The Oberoi Group. Next, I need to find the head office of The Oberoi Group. Based on the evidence, the head office of The Oberoi Group is in Delhi. <answer>Delhi</answer>.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: First I need to find out who Milhouse was named after. Based on the evidence, Milhouse was named after Richard Nixon. <answer>Richard Nixon</answer>.

Question: What nationality was James Henry Miller's wife?
Answer: First I need to find out who James Henry Miller's wife is. Based on the evidence, James Henry Miller's wife is named Peggy Seeger. Next, I need to find out the nationality of Peggy Seeger. Based on the evidence, Peggy Seeger is an American folksinger. <answer>American</answer>.

Question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Answer: First I need to find out what chemical Cadmium Chloride is slightly soluble in. Based on the evidence, Cadmium Chloride is slightly soluble in alcohol. <answer>alcohol</answer>.

Question: Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?
Answer: First I need to find out how many Grand Slam titles Henri Leconte has won. Based on the evidence, Henri Leconte has won 0 Grand Slam titles. Next, I need to find out how many Grand Slam titles Jonathan Stark has won. Based on the evidence, Jonathan Stark has won 2 Grand Slam titles. <answer>Jonathan Stark</answer>.

Question: Which genus of moth in the world's seventh-largest country contains only one species?
Answer: First I need to find out the world's seventh-largest country. Based on the evidence, the world's seventh-largest country is India. Next, I need to find out the genus of moth in India. Based on the evidence, Nepita is a genus of moth in India, Indogrammodes is a genus of moth in India. Next, I need to figure out the number of species in each genus. Based on the evidence, Nepita has 1 species and Indogrammodes has 1 species. <answer>Nepita,Indogrammodes</answer>.

Question: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?
Answer: First I need to find out who is the best kick boxer in the world. Based on the evidence, the best kick boxer in the world is Badr Hari. Next, I need to find out whether Badr Hari has been involved in controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring. Based on the evidence, Badr Hari has been involved in controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring. <answer>Badr Hari</answer>.

Question: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?
Answer: First I need to find the Dutch-Belgian television series that "House of Anubis" was based on. Based on the evidence, "House of Anubis" was based on "Het Huis Anubis". Next, I need to find out when "Het Huis Anubis" was first aired. Based on the evidence, "Het Huis Anubis" was first aired in September 2006. <answer>2006</answer>.

Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
Answer: First I need to find out what track the 2013 Liqui Moly Bathurst 12 Hour was staged on. Based on the evidence, the 2013 Liqui Moly Bathurst 12 Hour was staged on the Mount Panorama Circuit. Next, I need to find out the length of the track. Based on the evidence, the length of the track is 6.213 km. <answer>6.213 km</answer>.
"""

# ------------------------------------------------------------------------------------------------
# 2WikiMultiHopQA
# ------------------------------------------------------------------------------------------------
COT_EXAMPLES_2WIKI = f"""\
Example 1:
Question: Are director of film Move (1970 Film) and director of film M\u00e9diterran\u00e9e (1963 Film) from the same country?
Answer: Based on the evidence, the director of the film Move (1970 Film) is Stuart Rosenberg. Also, the director of the film M\u00e9diterran\u00e9e (1963 Film) is Jean-Daniel Pollet. The country of citizenship of Stuart Rosenberg is American. Also, the country of citizenship of Jean-Daniel Pollet is French. Because American and French aren't the same, the answer is no. <answer>no</answer>.

Example 2:
Question: What nationality is the director of film Borunbabur Bondhu?
Answer: Based on the evidence, the director of the film Borunbabur Bondhu is Anik Dutta. The country of citizenship of Anik Dutta is India. <answer>India</answer>.

Example 3:
Question: Where was the place of burial of the performer of song There Is So Much World To See?
Answer: Based on the evidence, the former of the song There Is So Much World To See is Elvis. The place of burial of Elvis is Graceland. <answer>Graceland</answer>.

Example 4:
Question: Why did the director of film The Light Of Western Stars (1930 Film) die?
Answer: Based on the evidence, the director of the film The Light Of Western Stars (1930 Film) is Otto Brower. The cause of death of Otto Brower is heart attack. <answer>heart attack</answer>.

Example 5:
Question: Which film came out first, The Love Route or Engal Aasan?
Answer: Based on the evidence, the release date of the film The Love Route is 1915. Also, the release date of the film Engal Aasan is 2009. Because 1915 is before 2009, the answer is The Love Route. <answer>The Love Route</answer>.

Example 6:
Question: Where was the director of film The Fascist born?
Answer: Based on the evidence, the director of the film The Fascist is Luciano Salce. The birthplace of Luciano Salce is Rome. <answer>Rome</answer>.

Example 7:
Question: Are Matraville Sports High School and Wabash High School both located in the same country?
Answer: Based on the evidence, the country that Matraville Sports High School is in is United States. Also, the country that Wabash High School is in is Australia. Because United States and Australia aren't the same, the answer is no. <answer>no</answer>.

Example 8:
Question: Which country the performer of song Soldier (Neil Young Song) is from?
Answer: Based on the evidence, the performer of the song Soldier is Neil Young. The country of citizenship of Neil Young is Canadian. <answer>Canadian</answer>.

Example 9:
Question: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Answer: Based on the evidence, the director of the film A Flame In My Heart is Alain Tanner. Also, the director of the film Butcher, Baker, Nightmare Maker is William Asher. The date of birth of Alain Tanner is 6 December 1929. Also, the date of birth of William Asher is August 8, 1921. Becuase 6 December 1929 is later than August 8, 1921, the answer is A Flame In My Heart. <answer>A Flame In My Heart</answer>.

Example 10:
Question: Which film has the director who died later, Aaranya Kandam or One Hundred Nails?
Answer: Based on the evidence, the director of the film Aaranya Kandam is J. Sasikumar. Also, the director of the film One Hundred Nails is Ermanno Olmi. The date of death of J. Sasikumar is 17 July 2014. The date of death of Ermanno Olmi is 7 May 2018. Because 7 May 2018 is later than 17 July 2014, the answer is One Hundred Nails. <answer>One Hundred Nails</answer>.
"""


# ------------------------------------------------------------------------------------------------
# MuSiQue
# ------------------------------------------------------------------------------------------------
COT_EXAMPLES_MSQ = """\
Example 1:
Question: Who was ordered to force a Tibetan assault into the region conquered by Yellow Tiger in the mid-17th century?
Answer: Based on the evidence, Yellow Tiger conquered the region of Sichuan in the mid-17th century. Based on the evidence, Ming general Qu Neng was ordered to force a Tibetan assault into Sichuan. <answer>Qu Neng</answer>.

Example 2:
Question: When did the publisher of Tetrisphere unveil their new systems?
Answer: Based on the evidence, Nintendo published Tetrisphere. Based on the evidence, Nintendo unveiled their new systems on October 18, 1985. <answer>October 18, 1985</answer>.

Example 3:
Question: Who is the composer of Rhapsody No. 1, named after and inspired by the county where Alfred Seaman was born?
Answer: Based on the evidence, Alfred Seaman was born in Norfolk. Based on the evidence, the Norfolk Rhapsodies were composed by Ralph Vaughan Williams. <answer>Ralph Vaughan Williams</answer>.

Example 4:
Question: What region is Qaleh Now-e Khaleseh in Mahdi Tajik's birth city located?
Answer: Based on the evidence, the birth city of Mahdi Tajik is Tehran. Based on the evidence, Qaleh Now-e Khaleseh is located in the Qaleh Now Rural District. <answer>Qaleh Now Rural District</answer>.

Example 5:
Question: What is Nasir Zaidi's birthplace the capital of?
Answer: Based on the evidence, Nasir Zaidi was born in the city of Karachi. Based on the evidence, Karachi is the capital of West Pakistan. <answer>West Pakistan</answer>.

Example 6:
Question: Who founded the publisher of Journal of Bisexuality?
Answer: Based on the evidence, the publisher of Journal of Bisexuality is Routledge. Based on the evidence, Routledge was founded by George Routledge. <answer>George Routledge</answer>.

Example 7:
Question: Where is the headquarters of located of the 48th Highlanders of the country which released the performer of As Seen Through Windows?
Answer: Based on the evidence, As Seen Through Windows was performed by the Bell Orchestre. Based on the evidence, Canada released the Bell Orchestre. Based on the evidence, the 48th Highlanders of Canada is headquartered in the Moss Park Armoury. <answer>Moss Park Armoury</answer>.

Example 8:
Question: Who is the spouse of the director of The Yellow Ticket?
Answer: Based on the evidence, the director of The Yellow Ticket is Sidney Lumet. Based on the evidence, the spouse of Sidney Lumet is Miriam Cooper. <answer>Miriam Cooper</answer>.

Example 9:
Question: When did the torch arrive in the country where Drunken Master was filmed?
Answer: Based on the evidence, Drunken Master was filmed in Hong Kong. Based on the evidence, the torch arrived in Hong Kong on May 2. <answer>May 2</answer>.

Example 10:
Question: Who is the mother of the singer of Catch You?
Answer: Based on the evidence, the singer of Catch You is Sophie Ellis-Bextor. Based on the evidence, the mother of Sophie Ellis-Bextor is Janet Ellis. <answer>Janet Ellis</answer>.
"""
