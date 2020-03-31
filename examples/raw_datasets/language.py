from spacy.lang.en import English
import spacy
from spacy.tokens import Span, Doc
from spacy.attrs import ORTH, LEMMA
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

def get_language(use_tokenizer_exceptions=True):
    """
    Retrieves a customized spaCy language
    :return:
    """

    language = spacy.load('en_core_sci_sm', disable=["tagger", "ner"])
    if use_tokenizer_exceptions:
        _add_tokenization_exceptions(language)
    language.add_pipe(_set_sentence_parse_exceptions, before='parser')

    Doc.set_extension('id', default=None, force=True)
    Span.set_extension('cui', default=None, force=True)
    return language

def _add_tokenization_exceptions(language):
    """
    Tons of tokenization exceptions for this dataset
    :param language:
    :return:
    """

    #N2C2 2019 and Share 2013 Concept Normalization
    language.tokenizer.add_special_case('empiricvancomycin', [{ORTH: "empiric"}, {ORTH: "vancomycin"}])
    language.tokenizer.add_special_case('dobutamine-MIBI', [{ORTH: 'dobutamine'}, {ORTH: '-'}, {ORTH: 'MIBI'}])
    language.tokenizer.add_special_case('cLVH', [{ORTH: 'c'}, {ORTH: 'LVH'}])
    language.tokenizer.add_special_case('UPEP/Beta', [{ORTH: 'UPEP'}, {ORTH: '/'}, {ORTH: 'Beta'}])
    language.tokenizer.add_special_case('constipation-related', [{ORTH: 'constipation'}, {ORTH: '-'}, {ORTH: 'related'}])
    language.tokenizer.add_special_case('anteriordysplasia', [{ORTH: 'anterior'}, {ORTH: 'dysplasia'}])
    language.tokenizer.add_special_case('F.', [{ORTH: 'F'}, {ORTH: '.'}])
    language.tokenizer.add_special_case('extrapleural-pleural', [{ORTH: 'extrapleural'}, {ORTH: '-'}, {ORTH: 'pleural'}])
    language.tokenizer.add_special_case('saphenous', [{ORTH: 'sap'}, {ORTH: 'henous'}])
    language.tokenizer.add_special_case('T97.5', [{ORTH: 'T'}, {ORTH: '97.5'}])
    language.tokenizer.add_special_case('P59', [{ORTH: 'P'}, {ORTH: '59'}])
    language.tokenizer.add_special_case('RR20', [{ORTH: 'RR'}, {ORTH: '20'}])
    language.tokenizer.add_special_case('Demerolprn', [{ORTH: 'Demerol'}, {ORTH: 'prn'}])
    language.tokenizer.add_special_case('Carboplatin-Taxolchemo', [{ORTH: 'Carboplatin'}, {ORTH: '-'}, {ORTH: 'Taxol'}, {ORTH: 'chemo'}])
    language.tokenizer.add_special_case('midepigastric', [{ORTH: 'mid'}, {ORTH: 'epigastric'}])
    language.tokenizer.add_special_case('BRBPR/melena', [{ORTH: 'BRBPR'}, {ORTH: '/'}, {ORTH: 'melena'}])
    language.tokenizer.add_special_case('CPAP+PS', [{ORTH: 'CPAP'}, {ORTH: '+'}, {ORTH: 'PS'}])
    language.tokenizer.add_special_case('medications', [{ORTH: 'medication'}, {ORTH: 's'}])
    language.tokenizer.add_special_case('mass-', [{ORTH: 'mass'}, {ORTH: '-'}])
    language.tokenizer.add_special_case('1.ampullary', [{ORTH: '1'}, {ORTH: '.'}, {ORTH: 'ampullary'}])
    language.tokenizer.add_special_case('membranes', [{ORTH: 'membrane'}, {ORTH: 's'}])
    language.tokenizer.add_special_case('SOBx', [{ORTH: 'SOB'}, {ORTH: 'x'}])
    language.tokenizer.add_special_case('Mass.', [{ORTH: 'Mass'}, {ORTH: '.'}])
    language.tokenizer.add_special_case('Atroventnebulizer', [{ORTH: 'Atrovent'}, {ORTH: 'nebulizer'}])
    language.tokenizer.add_special_case('PCO227', [{ORTH: 'PC02'}, {ORTH: '27'}])
    language.tokenizer.add_special_case('PCO227', [{ORTH: 'PC02'}, {ORTH: '27'}])
    language.tokenizer.add_special_case('MB&apos;s', [{ORTH: 'MB'}, {ORTH: '&apos;s'}])
    language.tokenizer.add_special_case('Q&apos;s', [{ORTH: 'Q'}, {ORTH: '&apos;s'}])
    language.tokenizer.add_special_case('predischarge', [{ORTH: 'pre'}, {ORTH: 'discharge'}])
    language.tokenizer.add_special_case('1.  Diabetes mellitus type 2.', [{ORTH: '1.  '}, {ORTH: 'Diabetes mellitus type 2'}, {ORTH: '.'}])

    #N2C2 2018 NER
    language.tokenizer.add_special_case('ons', [{ORTH: 'on'}, {ORTH: 's'}])
    language.tokenizer.add_special_case('DAILY16', [{ORTH: 'DAILY'}, {ORTH: '16'}])
    language.tokenizer.add_special_case('2uRBCs,', [{ORTH: '2u'}, {ORTH: 'RBCs'},{ORTH: ','}])
    language.tokenizer.add_special_case('1.amlodipine', [{ORTH: '1'}, {ORTH: '.'},{ORTH: 'amlodipine'}])
    language.tokenizer.add_special_case('2.fexofenadine', [{ORTH: '2'}, {ORTH: '.'},{ORTH: 'fexofenadine'}])
    language.tokenizer.add_special_case('3.levothyroxine', [{ORTH: '3'}, {ORTH: '.'},{ORTH: 'levothyroxine'}])
    language.tokenizer.add_special_case('4.omeprazole', [{ORTH: '4'}, {ORTH: '.'},{ORTH: 'omeprazole'}])
    language.tokenizer.add_special_case('5.multivitamin', [{ORTH: '5'}, {ORTH: '.'},{ORTH: 'multivitamin'}])
    language.tokenizer.add_special_case('6.tiotropium', [{ORTH: '6'}, {ORTH: '.'},{ORTH: 'tiotropium'}])
    language.tokenizer.add_special_case('7.atorvastatin', [{ORTH: '7'}, {ORTH: '.'},{ORTH: 'atorvastatin'}])
    language.tokenizer.add_special_case('8.docusate', [{ORTH: '8'}, {ORTH: '.'},{ORTH: 'docusate'}])
    language.tokenizer.add_special_case('9.dofetilide', [{ORTH: '9'}, {ORTH: '.'},{ORTH: 'dofetilide'}])
    language.tokenizer.add_special_case('10.albuterol', [{ORTH: '10'}, {ORTH: '.'},{ORTH: 'albuterol'}])
    language.tokenizer.add_special_case('11.cholecalciferol', [{ORTH: '11'}, {ORTH: '.'},{ORTH: 'cholecalciferol'}])
    language.tokenizer.add_special_case('12.fluticasone', [{ORTH: '12'}, {ORTH: '.'},{ORTH: 'fluticasone'}])
    language.tokenizer.add_special_case('13.morphine', [{ORTH: '13'}, {ORTH: '.'},{ORTH: 'morphine'}])
    language.tokenizer.add_special_case('14.morphine', [{ORTH: '14'}, {ORTH: '.'},{ORTH: 'morphine'}])
    language.tokenizer.add_special_case('15.calcium', [{ORTH: '15'}, {ORTH: '.'},{ORTH: 'calcium'}])
    language.tokenizer.add_special_case('16.warfarin', [{ORTH: '16'}, {ORTH: '.'},{ORTH: 'warfarin'}])
    language.tokenizer.add_special_case('17.warfarin', [{ORTH: '17'}, {ORTH: '.'},{ORTH: 'warfarin'}])
    language.tokenizer.add_special_case('18.Epogen', [{ORTH: '18'}, {ORTH: '.'},{ORTH: 'Epogen'}])
    language.tokenizer.add_special_case('19.guaifenesin', [{ORTH: '19'}, {ORTH: '.'},{ORTH: 'guaifenesin'}])
    language.tokenizer.add_special_case('20.bumetanide', [{ORTH: '20'}, {ORTH: '.'},{ORTH: 'bumetanide'}])
    language.tokenizer.add_special_case('21.prednisone', [{ORTH: '21'}, {ORTH: '.'},{ORTH: 'prednisone'}])
    language.tokenizer.add_special_case('22.ferrous', [{ORTH: '22'}, {ORTH: '.'},{ORTH: 'ferrous'}])
    language.tokenizer.add_special_case('23.spironolactone', [{ORTH: '23'}, {ORTH: '.'},{ORTH: 'spironolactone'}])
    language.tokenizer.add_special_case('1.lasix', [{ORTH: '1'}, {ORTH: '.'},{ORTH: 'lasix'}])
    language.tokenizer.add_special_case('6.lasix', [{ORTH: '6'}, {ORTH: '.'},{ORTH: 'lasix'}])
    language.tokenizer.add_special_case('10.citalopram', [{ORTH: '10'}, {ORTH: '.'},{ORTH: 'citalopram'}])
    language.tokenizer.add_special_case('2.haloperidol', [{ORTH: '2'}, {ORTH: '.'},{ORTH: 'haloperidol'}])
    language.tokenizer.add_special_case('4.tiotropium', [{ORTH: '4'}, {ORTH: '.'},{ORTH: 'tiotropium'}])
    language.tokenizer.add_special_case('8.omeprazole', [{ORTH: '8'}, {ORTH: '.'},{ORTH: 'omeprazole'}])
    language.tokenizer.add_special_case('3.tamsulosin', [{ORTH: '3'}, {ORTH: '.'},{ORTH: 'tamsulosin'}])
    #language.tokenizer.add_special_case('.atorvastatin', [{ORTH:'.'},{ORTH: 'atorvastatin'}])
    language.tokenizer.add_special_case('5.atorvastatin', [{ORTH: '5'}, {ORTH: '.'},{ORTH: 'atorvastatin'}])
    language.tokenizer.add_special_case('9.aspirin', [{ORTH: '5'}, {ORTH: '.'},{ORTH: 'atorvastatin'}])
    language.tokenizer.add_special_case('10.citalopram', [{ORTH: '10'}, {ORTH: '.'},{ORTH: 'citalopram'}])
    language.tokenizer.add_special_case('1.fluticasone-salmeterol', [{ORTH: '1'}, {ORTH: '.'},{ORTH: 'fluticasone'},{ORTH:'-'}, {ORTH: 'salmeterol'}])
    language.tokenizer.add_special_case('6.lisinopril', [{ORTH: '6'}, {ORTH: '.'}, {ORTH: 'lisinopril'}])
    language.tokenizer.add_special_case('7.senna', [{ORTH: '7'}, {ORTH: '.'}, {ORTH: 'senna'}])
    language.tokenizer.add_special_case('hours).', [{ORTH: 'hours'}, {ORTH: ')'}, {ORTH: '.'}])
    language.tokenizer.add_special_case('.Talon', [{ORTH: '.'}, {ORTH: 'Talon'}])

    language.tokenizer.add_special_case('RR<', [{ORTH: 'RR'}, {ORTH: '<'}])
    language.tokenizer.add_special_case('(2', [{ORTH: '('}, {ORTH: '2'}])
    language.tokenizer.add_special_case('IDDM:', [{ORTH: 'ID'}, {ORTH: 'DM'}, {ORTH: ':'}])
    language.tokenizer.add_special_case('@HS,tramadol', [{ORTH: '@'}, {ORTH: 'HS'},{ORTH: ','}, {ORTH: 'tramadol'}])
    language.tokenizer.add_special_case('1-2Lnc', [{ORTH: '1-2L'}, {ORTH: 'nc'}])
    language.tokenizer.add_special_case('withantibiotic', [{ORTH: 'with'}, {ORTH: 'antibiotic'}])

    language.tokenizer.add_special_case('startingKeppra,', [{ORTH: 'starting'}, {ORTH: 'Keppra'}])
    language.tokenizer.add_special_case('Warfarin5', [{ORTH: 'Warfarin'}, {ORTH: '5'}])
    language.tokenizer.add_special_case('IDDM', [{ORTH: 'I'}, {ORTH: 'DDM'}])
    language.tokenizer.add_special_case('1u', [{ORTH: '1'}, {ORTH: 'u'}])
    language.tokenizer.add_special_case('6U', [{ORTH: '6'}, {ORTH: 'U'}])
    language.tokenizer.add_special_case('HSQ', [{ORTH: 'H'}, {ORTH: 'SQ'}])
    language.tokenizer.add_special_case('GD20', [{ORTH: 'GD'}, {ORTH: '20'}])
    language.tokenizer.add_special_case('FAFA', [{ORTH: 'FA'}, {ORTH: 'FA'}])
    language.tokenizer.add_special_case('FACB', [{ORTH: 'FA'}, {ORTH: 'CB'}])
    language.tokenizer.add_special_case('O3CB', [{ORTH: 'O3'}, {ORTH: 'CB'}])
    language.tokenizer.add_special_case('O3FA', [{ORTH: '03'}, {ORTH: 'FA'}])
    language.tokenizer.add_special_case('PND5', [{ORTH: 'PND'}, {ORTH: '5'}])
    language.tokenizer.add_special_case('PND60:', [{ORTH: 'PND'}, {ORTH: '60'}, {ORTH: ':'}])
    language.tokenizer.add_special_case('mice/treatment)', [{ORTH: 'mice'}, {ORTH: '/'}, {ORTH: 'treatment'}, {ORTH: ')'}])


    #TAC 2018
    language.tokenizer.add_special_case('Kunmingmouse', [{ORTH: 'Kunming'}, {ORTH: 'mouse'}])
    language.tokenizer.add_special_case('24h', [{ORTH: '24'}, {ORTH: 'h'}])
    language.tokenizer.add_special_case('72h', [{ORTH: '72'}, {ORTH: 'h'}])
    language.tokenizer.add_special_case('[15N5]8-oxodG', [{ORTH: '[15N5]'}, {ORTH: '8-oxodG'}])
    language.tokenizer.add_special_case('ratswerepermitted', [{ORTH: 'rats'}, {ORTH: 'were'} ,{ORTH: 'permitted'}])
    language.tokenizer.add_special_case('mgTi', [{ORTH: 'mg'}, {ORTH: 'Ti'}])
    language.tokenizer.add_special_case('ND60', [{ORTH: 'ND'}, {ORTH: '60'}])
    # language.tokenizer.add_special_case('PND30–35', [{ORTH: 'PND'}, {ORTH: '30–35'}])
    language.tokenizer.add_special_case('198Au', [{ORTH: '198'}, {ORTH: 'Au'}])
    language.tokenizer.add_special_case('8weeks,', [{ORTH: '8'}, {ORTH: 'weeks'}, {ORTH:','}])
    language.tokenizer.add_special_case('weeks:55', [{ORTH: 'weeks'}, {ORTH: ':'}, {ORTH:'55'}])
    language.tokenizer.add_special_case('ininfected', [{ORTH: 'in'}, {ORTH: 'infected'}])
    language.tokenizer.add_special_case('15days.', [{ORTH: '15'}, {ORTH: 'days'},{ORTH: '.'}])
    language.tokenizer.add_special_case('GD18', [{ORTH: 'GD'},{ORTH: '18'}])
    language.tokenizer.add_special_case('day).', [{ORTH: 'day'},{ORTH: ')'}, {ORTH: '.'}])
    language.tokenizer.add_special_case('x11days).', [{ORTH: 'x11'},{ORTH: 'days'}, {ORTH: ')'},{ORTH: '.'}])
    language.tokenizer.add_special_case('4.5hours', [{ORTH: '4.5'}, {ORTH: 'hours'}])
    language.tokenizer.add_special_case('0.5mg', [{ORTH: '0.5'}, {ORTH: 'mg'}])


    #N2C2 2010
    language.tokenizer.add_special_case('periprosthetic', [{ORTH: 'peri'}, {ORTH: 'prosthetic'}])
    language.tokenizer.add_special_case('MIER', [{ORTH: 'MI'}, {ORTH: 'ER'}])

    #END 2017
    language.tokenizer.add_special_case('PeripheralPeripheral', [{ORTH: 'Peripheral'}, {ORTH: 'Peripheral'}])
    language.tokenizer.add_special_case('SeriousSerious', [{ORTH: 'Serious'}, {ORTH: 'Serious'}])
    language.tokenizer.add_special_case('ADC-CD30', [{ORTH: 'ADC-CD'}, {ORTH: '30'}])
    language.tokenizer.add_special_case('MCC-DM1', [{ORTH: 'MCC-DM'}, {ORTH: '1'}])
    language.tokenizer.add_special_case('syndrome[see', [{ORTH: 'syndrome'}, {ORTH: '['}, {ORTH: 'see'}])
    language.tokenizer.add_special_case('5.1Anaphylaxis', [{ORTH: '5.1'}, {ORTH: 'Anaphylaxis'}])
    language.tokenizer.add_special_case('HIGHLIGHTSPEGINTRON', [{ORTH: 'HIGHLIGHTS'}, {ORTH: 'PEGINTRON'}])
    language.tokenizer.add_special_case('HIGHLIGHTSRibavirin', [{ORTH: 'HIGHLIGHTS'}, {ORTH: 'Ribavirin'}])
    language.tokenizer.add_special_case('COPEGUS[see', [{ORTH: 'COPEGUS'}, {ORTH: '[see'}])



    #I2B2 2014
    language.tokenizer.add_special_case('FAT', [{ORTH: 'F'}, {ORTH: 'A'}, {ORTH: 'T'}])
    language.tokenizer.add_special_case('TTS', [{ORTH: 'T'}, {ORTH: 'T'}, {ORTH: 'S'}])
    language.tokenizer.add_special_case('STTh', [{ORTH: 'S'}, {ORTH: 'T'}, {ORTH: 'Th'}])
    language.tokenizer.add_special_case('TThSa', [{ORTH: 'T'}, {ORTH: 'h'}, {ORTH: 'Sa'}])
    language.tokenizer.add_special_case('MWFS', [{ORTH: 'M'}, {ORTH: 'W'}, {ORTH: 'F'}, {ORTH: 'S'}])
    language.tokenizer.add_special_case('MWF', [{ORTH: 'M'}, {ORTH: 'W'}, {ORTH: 'F'}])
    language.tokenizer.add_special_case('ThisRoberta', [{ORTH: 'This'}, {ORTH: 'Roberta'}])
    language.tokenizer.add_special_case('GambiaHome', [{ORTH: 'Gambia'}, {ORTH: 'Home'}])
    language.tokenizer.add_special_case('SupervisorSupport', [{ORTH: 'Supervisor'}, {ORTH: 'Support'}])
    language.tokenizer.add_special_case('inhartsville', [{ORTH: 'in'}, {ORTH: 'hartsville'}])
    language.tokenizer.add_special_case('ELLENMRN:', [{ORTH: 'ELLEN'}, {ORTH: 'MRN:'}])
    language.tokenizer.add_special_case('0.411/29/2088', [{ORTH: '0.4'}, {ORTH: '11/29/2088'}])
    language.tokenizer.add_special_case('past11/29/2088', [{ORTH: 'past'}, {ORTH: '11/29/2088'}])
    language.tokenizer.add_special_case('Hospital0021', [{ORTH: 'Hospital'}, {ORTH: '0021'}])
    language.tokenizer.add_special_case('HospitalAdmission', [{ORTH: 'Hospital'}, {ORTH: 'Admission'}])
    language.tokenizer.add_special_case('AvenueKigali,', [{ORTH: 'Avenue'}, {ORTH: 'Kigali,'}])
    language.tokenizer.add_special_case('47798497-045-1949', [{ORTH: '47798'}, {ORTH: '497-045-1949'}])
    language.tokenizer.add_special_case('.02/23/2077:', [{ORTH: '.'}, {ORTH: '02/23/2077:'}])
    language.tokenizer.add_special_case('34712RadiologyExam', [{ORTH: '34712'}, {ORTH: 'RadiologyExam'}])
    language.tokenizer.add_special_case('3041038MARY', [{ORTH: '3041038'}, {ORTH: 'MARY'}])
    language.tokenizer.add_special_case('PLAN88F', [{ORTH: 'PLAN'}, {ORTH: '88'}, {ORTH: 'F'}])
    language.tokenizer.add_special_case('~2112Hypothyroidism', [{ORTH: '~'}, {ORTH: '2112'}, {ORTH: 'Hypothyroidism'}])
    language.tokenizer.add_special_case('97198841PGH', [{ORTH: '97198841'}, {ORTH: 'PGH'}])
    language.tokenizer.add_special_case('5694653MEDIQUIK', [{ORTH: '5694653'}, {ORTH: 'MEDIQUIK'}])
    language.tokenizer.add_special_case('0083716SNH', [{ORTH: '0083716'}, {ORTH: 'SNH'}])
    language.tokenizer.add_special_case('20626842267', [{ORTH: '2062'}, {ORTH: '6842267'}])
    language.tokenizer.add_special_case('0370149RSC', [{ORTH: '0370149'}, {ORTH: 'RSC'}])
    language.tokenizer.add_special_case('4832978HOB', [{ORTH: '4832978'}, {ORTH: 'HOB'}])
    language.tokenizer.add_special_case('0907307PCC', [{ORTH: '0907307'}, {ORTH: 'PCC'}])
    language.tokenizer.add_special_case('LittletonColonoscopy', [{ORTH: 'Littleton'}, {ORTH: 'Colonoscopy'}])
    language.tokenizer.add_special_case('34674TSH', [{ORTH: '34674'}, {ORTH: 'TSH'}])
    language.tokenizer.add_special_case('b93D', [{ORTH: 'b'}, {ORTH: '93'}, {ORTH: 'D'}])
    language.tokenizer.add_special_case('due22D', [{ORTH: 'due'}, {ORTH: '22'}, {ORTH: 'D'}])
    language.tokenizer.add_special_case('33182William', [{ORTH: '33182'}, {ORTH: 'William'}])

    #French 2014 NER
    language.tokenizer.add_special_case('postopératoires', [{ORTH: 'post'}, {ORTH: 'opératoires'}])


    custom_infixes = [r'\d+\.\d+','[P|p]RBCs?','cap|CAP','qhs|QHS|mg','tab|TAB|BPA', 'BB', 'yo','ASA','gtt|GTT','iv|IV', 'FFP',
                      'inh|INH', 'pf|PF', 'bid|BID|PND','prn|PRN','puffs?',r'\dL',
                      'QD|qd','Q?(AM|PM)','O2','MWF',r'q\d+', 'HS' , 'ye423zc',
                      '-|;|:|–|#|<|{|}', r'\-' ] + [r"\\", "/", "%", r"\+", r"\,", r"\(", r"\)", r"\.", r"\d\d/\d\d/\d\d\d\d", r"\d\d\d\d\d\d\d", r"\^"]
    language.tokenizer.infix_finditer = compile_infix_regex(tuple(list(language.Defaults.infixes) + custom_infixes)).finditer
    #language.tokenizer.infix_finditer = compile_infix_regex(custom_infixes).finditer
    language.tokenizer.prefix_search = compile_prefix_regex(tuple(list(language.Defaults.prefixes)
                                                                  + ['-|:|_+', '/', '~','x|X',r'\dL', 'O2','VN',"Coumadin",
                                                                     "HIGHLIGHTS","PROMPTCARE", "VISUDYNE","weeks", "week", 'ASA', 'pap',
                                                                     "ZT", "BaP", "PND|BID", "BPA", "GD", 'BB', 'PBS',
                                                                     "days", "day","Kx", 'mg', r"\d\.'", "Results", "RoeID", "at", r"\d/\d\d\d\d"])).search
    language.tokenizer.suffix_search = compile_suffix_regex(tuple(list(language.Defaults.suffixes)
                                                                  + ['weeks|minutes|day|days|hours|year',
                                                                     'Au',r"\[see", 'induced|IMA|HBMC|mice|MARY|SLO|RHM|solutions|—and|for|III|FINAL|The|Scattered|Intern|Left|Emergency|Staff|Chief',
                                                                     'μl|μL|μg/L|AGH|HCC|RHN|MC|yM|GMH|Code|Hyperlipidemia|Adenomatous|greater|Drug|MEDIQUIK|and|Date|Procedure|Problems|Ordering|CLOSURE|Total|Status',
                                                                     ':|_+',r"\.\w+", 'U|D', "Mouse", r"\d\d/\d\d/\d\d\d\d", r"\d\d/\d\d"])).search
    #exit()
    #print(list(language.Defaults.infixes))


def _set_sentence_parse_exceptions(doc):
    for token in doc[:-1]:
        # if token.text == '\n':
        #     doc[token.i+1].is_sent_start = True
        if token.text == 'B-3':
            doc[token.i-1].is_sent_start = False
            doc[token.i].is_sent_start = False
        if token.text == 'Troponin':
            doc[token.i+1].is_sent_start = False
            doc[token.i-1].is_sent_start = False
        if token.text == 'p.o' or token.text == 'po' or token.text == '.q' or token.text == 'VIT' and len(doc)>2:
            doc[token.i+2].is_sent_start = False
        if token.text == '12.7' and len(doc)>token.i + 2:
            doc[token.i + 2].is_sent_start = False
    return doc
