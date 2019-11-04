# CHAPTER 4
# Web Forms

# WTForms standard HTML fields
# 
# Field : type Description
# StringField : Text field
# TextAreaField : Multiple-line text field
# PasswordField : Password text field
# HiddenField : Hidden text field
# DateField : Text field that accepts a datetime.date value in a given format
# DateTimeField : Text field that accepts a datetime.datetime value in a given format
# IntegerField : Text field that accepts an integer value
# DecimalField : Text field that accepts a decimal.Decimal value
# FloatField : Text field that accepts a floating-point value
# BooleanField : Checkbox with True and False values
# RadioField : List of radio buttons
# SelectField : Drop-down list of choices
# SelectMultipleField : Drop-down list of choices with multiple selection
# FileField : File upload field
# SubmitField : Form submission button
# FormField : Embed a form as a field in a container form
# FieldList : List of fields of a given type

# WTForms validators
# 
# Validator : Description
# Email : Validates an email address
# EqualTo : Compares the values of two fields; useful when requesting a password to be entered twice for confirmation
# IPAddress : Validates an IPv4 network address
# Length : Validates the length of the string entered
# NumberRange : Validates that the value entered is within a numeric range
# Optional : Allows empty input on the field, skipping additional validators
# Required : Validates that the field contains data
# Regexp : Validates the input against a regular expression
# URL : Validates a URL
# AnyOf : Validates that the input is one of a list of possible values
# NoneOf : Validates that the input is none of a list of possible values

from flask import Flask, render_template, send_file
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
# from wtforms.validators import DataRequired
from wtforms.validators import Required

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
moment = Moment(app)

class NameForm(FlaskForm):
    name = StringField('What is your name?', validators=[Required()])
    submit = SubmitField('Submit')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        form.name.data = ''
    return render_template('index.html', form=form, name=name)

@app.route('/sentiment-analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    df0 = [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'positive',
        'negative',
        'positive',
        'positive',
        'positive',
        ]

    show_table = 10
    text_limit = 200

    neg_count = df0.count('negative')
    pos_count = df0.count('positive')

    data = {
        'SentimentText' : [
            "first think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie 10 8 years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give 10/10!",
            "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one. don't know exactly plagued movie. never thought I'd say this, want 15 minutes fame back.Script, Direction, can't say. recognized stable actors (the usual suspects), thought Herbert Marshall class addition sat good cheesy flick. Boy, wrong. Dullsville.My favorite parts: \"\"office girl\"\" makes 029 keypunch puts cards 087 sorter. LOL @ \"\"the computer\"\". I'd like someone identify next device - 477 ? It's even dinosaur's time.And dinosaurs don't much time waste.",
            "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family. moved new house, seem happy. pet cemetery behind house. Creed's new neighbor Jud (played Fred Gwyne) explains burial ground behind pet cemetery. burial ground pure evil. Jud tells Louis Creed bury human (or kind pet) burial ground, would come back life. problem, come back, person, they're evil. Soon Jud explains everything Pet Sematary, everything starts go hell. wont explain anymore don't want give away main parts film. acting Pet Sematary pretty good, needed little bit work. story one main parts movie, mainly original gripping. film features lots make-up effects make movie way eerie, frightening. One basic reasons movie sent chills back, fact make-up effects. one character film truly freaky. character \"\"Zelda.\"\" particular character pops film three times precise. Zelda Rachel Creed's sister passed away years before, Rachel still haunted her. first time Zelda appears movie isn't generally scary isn't talking anything, second time worst, honest, second time scares living **** me. absolutely nothing wrong movie, almost perfect. Pet Sematary delivers great scares, pretty good acting, first rate plot, mesmerizing make-up. truly one favorite horror films time. 10 10.",
            "watched horrid thing TV. Needless say one movies watch see much worse get. Frankly, don't know much lower bar go. The characters composed one lame stereo-type another, obvious attempt creating another \"\"Bad News Bears\"\" embarrassing say least.I seen prized turkeys time, reason list since \"\"Numero Uno\"\".Let put way, watched Vanilla Ice movie, bad funny. This...this...is even good.",
            "truly enjoyed film. acting terrific plot. Jeff Combs talent recognized for. part flick would change ending. death creature far gruesome Sci Fi Channel.There interesting religious messages film. Jeff Combs obviously played Messiah figure creature (or shark prefer) represented anti-Chirst. particularly frightening scenes 'end world feel'. noticed third viewing classic creature feature. know many people won't get references Christianity, watch close you'll get it.",
            "memory \"\"The Last Hunt\"\" stuck since saw 1956 13. movie far ahead others time addressed treatment natives, environment, ever present contrast short long term effects greed. relevant today 1956, cinemagraphic discussion utmost depth relevance. top setting beautiful cinematography excellent. memory movie end days.",
            "Shakespeare fan, appreciate Ken Branagh done bring Shakespeare back new generation viewers. However, movie falls short conveying overall intentions play ridiculous musical sequences. Add Alicia Silverstone's stumbling dialogue (reminiscent Keanu Reeves Much Ado Nothing) poorly cast roles, equals excruciating endurance viewing.",
            "privilege watching Scarface big screen beautifully restored 35mm print honor 20th anniversary films release. great see big screen much lost television sets overall largesse project cannot emphasized enough. Scarface remake classic rags riches depths hell story featuring Al Pacino Cuban drug lord Tony Montana. version, Tony comes America Cuban boat people immigration wave late 1970s, early 1980s. Tony cohorts quickly get green cards offing political figure Tent City brief stay Cuban restaurant; Tony launched horrific path towards total destruction. Many characters movie played skilled manner enjoyable watch forgot little film last twenty years. Robert Loggia Tony's patron, Frank Lopez wonderful. character flawed trusting, Tony quickly figures out, soft. Lopez's right hand, Omar Suarez portrayed one greatest actors, F. Murray Abraham (Amadeus.) Suarez ultimate toady anything Frank; like mind own. Tony quickly sees constantly battles Suarez, really sees minor problem get way top. character always comes back played perfectly Mel Bernstein, audaciously corrupt Miami Narcotics detective played Harris Yulin (Training Day.) Mel, without guilt extorts great sums money form sides drug industry. plays Tony Frank catches scene marks exit film Frank Mel. priceless hear Frank asking Mel intercede, Tony kill hear Mel reply, `It's tree Frank, you're sitting it.' man Frank paying protection!Tony's rise meteoric matched speed intensity quick crash burn. offing Frank taking wife business Tony's greed takes never seem get enough. Tony plunges deeper world drugs, greed inability trust eventually kills best friend sister fallen love married. sets ending Tony's compound stormed army supplier feels betrayed Tony would go political assassination ordered. stems form compassionate moment Tony refused accomplice murder would involved victim's wife children.All great depiction 1980s excess cocaine culture. DePalma nice job holding together one fastest moving three hour movies around. violence extremely graphic contains scenes forever etched viewers mind, particularly gruesome chainsaw seen, two point blank shots head entire bloody melee ends movie. highly recommended stylistically done film squeamish, need upbeat endings potential sequels; DePalma let fly right here.",
            "real classic. shipload sailors trying get towns daughters fathers go extremes deter sailors attempts. maidens cry aid results dispatch \"\"Rape Squad\"\". cult film waiting happen!",
            "Serials short subjects originally shown theaters conjunction feature film related pulp magazine serialized fiction. Known \"\"chapter plays,\"\" extended motion pictures broken number segments called \"\"chapters\"\" \"\"episodes.\"\" chapter would screened theater one week. serial would end cliffhanger, hero heroine would find latest perilous situation could escape.The audience would return next week find hero heroine would escape battle villain again. Serials especially popular children, many children first half 20th century, typical Saturday movies included chapter least one serial, along animated cartoons, newsreels, two feature films.The golden age serials 1936-1945. one best era.Zorro seen many films, Reed Hadley (\"\"Racket Squad\"\", Undying Brain) excellent role.The action constant, led chapter chapter ultimate end find identity evildoer.Zorro triumphs, always does."
        ]
    }

    # full_dataset = data['SentimentText'].values[0:show_table]
    full_dataset = data['SentimentText'][0:show_table]
    text_dataset = [_[:text_limit] + '...' if len(_)>text_limit else _ for _ in full_dataset]
    text_more = [_ if len(_)>text_limit else '' for _ in full_dataset]
    return render_template(
        'index_sentimental_analysis.html',
        number =  range(len(df0)),
        output = df0,
        text_dataset = text_dataset,
        text_more = text_more,
        text_limit = text_limit,
        neg_count = neg_count,
        pos_count = pos_count
    )

@app.route('/getDatasetCSV')
def plot_csv():
    return send_file(
        'dataset.csv',
        mimetype='text/csv',
        attachment_filename='data set.csv',
        as_attachment=True
    )

# example 4-5
from flask import Flask, render_template, session, redirect, url_for

@app.route('/test_post', methods=['GET', 'POST'])
def diff_index():
    form = NameForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        return redirect(url_for('diff_index'))
    return render_template('index.html', form=form, name=session.get('name'))
    # return render_template('index.html', form=form, name=name)

# example 4-6
from flask import Flask, render_template, session, redirect, url_for, flash
@app.route('/message_flash', methods=['GET', 'POST'])
def message_flash():
    form = NameForm()
    if form.validate_on_submit():
        old_name = session.get('name')
        if old_name is not None and old_name != form.name.data:
            flash('Looks like you have changed your name!')
        session['name'] = form.name.data
        form.name.data = ''
        return redirect(url_for('message_flash'))
    return render_template('index.html', form = form, name = session.get('name'))

@app.route('/about-us', methods=['GET', 'POST'])
def about_us():
    return render_template('about_us.html')

@app.route('/contact-us', methods=['GET', 'POST'])
def contact_us():
    return render_template('contact_us.html')

@app.route('/our-project', methods=['GET', 'POST'])
def our_project():
    return render_template('our_project.html')

# text frequency
from flask import Flask, render_template,request
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
import plotly.express as px

data = pd.read_csv('dataset.csv', encoding = "ISO-8859-1")
#top 20 words before removing stopwords
data['Sentiment'] = data['Sentiment'].replace(0,'negative')
data['Sentiment'] = data['Sentiment'].replace(1,'positive')
data = data[data.Sentiment != "positive"]
data['SentimentText'] = data['SentimentText'].apply((lambda x: re.sub('br ','',x)))

#top 20 words after removing stopwords
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(data['SentimentText'], 20)
df2 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])

#top 20 bigrams after removing stopwords
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(data['SentimentText'], 20)
df3 = pd.DataFrame(common_words, columns = ['SentimentText' , 'count'])

@app.route('/top-20-words')
def top_20_words():
    feature = 'Top 20 Words'
    bar = create_plot(feature)
    data1 = [go.Bar(x=df2['SentimentText'], y=df2['count'])]
    graphJSON = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index_text_frequency.html', plot=bar, graphJSON=graphJSON)

def create_plot(feature):
    if feature == 'Top 20 Words':
        #fig2 = px.bar(df2, x='SentimentText', y='count') #top words plot
        #fig2.show()
        data1 = [go.Bar(x=df2['SentimentText'], y=df2['count'])]
        graphJSON = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)
    elif feature == 'Top 20 Bigrams' :
        #fig2 = px.bar(df3, x='SentimentText', y='count') #top bigrams plot
        #fig2.show()
        data2 = [go.Bar(x=df3['SentimentText'], y=df3['count'])]
        graphJSON = json.dumps(data2, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():
    feature = request.args['selected']
    graphJSON = create_plot(feature)
    return graphJSON

if __name__ == '__main__':
	app.run(debug=True)