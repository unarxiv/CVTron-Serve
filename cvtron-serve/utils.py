import json
import requests
from jinja2 import Template

def renderTemplate(taskId):
    temp = Template("""
        <p>Your training task {{taskId}} has done succesfully.</p>
        <p>Please download your model with the button below.</p>
        <table border="0" cellpadding="0" cellspacing="0" class="btn btn-primary">
            <tbody>
                <tr>
                    <td align="left">
                        <table border="0" cellpadding="0" cellspacing="0">
                            <tbody>
                                <tr>
                                    <td> <a href="{{link}}" target="_blank">Download</a> </td>
                                </tr>
                            </tbody>
                        </table>
                    </td>
                </tr>
            </tbody>
        </table>
        <p>For logs, you can login to your console at our <a href="https://label.cvtron.xyz">Labeling Toolkit</a></p>
    """)
    link = 'http://134.175.1.246/static/'+taskId + '/' + taskId + '.zip'
    return temp.render(taskId=taskId, link=link)

def inform (taskId, emailAddr):
   host = 'http://email.cloud.zhitantech.com/email'
   headers = {'Content-Type': 'application/json'}
   payload = {
       'from': 'noreply',
       'to': emailAddr,
       'content': renderTemplate(taskId),
       'title': 'Your Training Has Finished Successfully'
   }
   r = requests.post(host, headers=headers, data=json.dumps(payload))
   print(r)
   print(r.text)

#inform('1234','407718364@qq.com')
