---
layout: post
title:  "My first blog"
date:   2024-01-23 21:58:25 +0900
categories: jekyll update
---
<mark>드디어 성공한 첫 블로그(Eunoia of IM)</mark>

<br>
오늘 하루, github 블로그를 개설하기 위해 많은 시간을 쏟아부었다....
체감상 3시간은 끙끙대며 작성한 것 같은데 사실 블로그 생성만 한다면 오래 안 걸린다.... 블로그의 테마 적용이 오래 걸릴 뿐! 그래도 마음에 드는 테마를 적용해서 정말 기쁘다.


<br>
<br>

첫 게시글은 github 블로그를 개설하며 사용한 코드와 기억해야 할 사항을 간단히 적어보겠다

<br>
<br>

>**블로그 개설 및 테마 적용 단계**

1. username.github.io 형식의 이름을 가진 새로운 repositorie 생성하기
2.  clone 사용하여 로컬과 github 연결시켜주기
3. git add -> git commit -> git push 과정으로 github에 새롭게 올리기
4. 테마 적용 전, ruby와 jekyll 다운하기
-> **_버전을 조심하며 다운!!_**
5. bundle 사용하여 테마를 미리 적용시킨 다음, 잘 적용되었는지 확인하고 git add -> git commit -> git push 로 github에도 적용시켜주기

<br>
<br>
이렇게 보면 매우 간단해 보이지만, 실제로는 많은 에러들이 발생하여 여러 블로그들을 참고하며 에러를 수정했다. 에러가 뜨는 문장을 google에 그대로 검색하면 많은 천사분들이 도와주시고 계시기 때문에 혼자서도 해결 할 수 있다!ㅎㅎ 

<br>
<br>

하지만, 많은 에러 중에서도 유독 ruby, jekyll 버전 에러에 대한 글이 없었는데 **어떤 에러인지 감이 안 잡히는 에러가 뜬다면 ruby, jekyll 버전부터 다시 업그레이드/변경 해보자!**

<br>
<br>

>**블로그 업로드 방법**
: cmd의 username.github.io 폴더로 접근


git add .
{: .message }
git commit -m "Initial commit"
{: .message }
it push
{: .message }
<br>

_post 파일에 블로그 글을 넣은 후, 위 세 단계를 진행하면 된다.
블로그 업로드뿐만 아니라 블로그에 다른 변경 사항이 생겼을 때 위의 세 단계를 꼭 진행하자! github를 확인해보면, 진행 사항과 성공 여부와 시간까지 확인할 수 있다. 초록색 체크가 뜨면 성공이다.
<br>
<br>

![Alt text](%EC%A7%84%ED%96%89%EC%82%AC%ED%95%AD.PNG)


<br>
<br>

**블로그 이름**
: Eunoia of IM

Eunoia : 아름다운 생각 이라는 단어의 스펠링이 내 이름과 비슷하여 눈여겨보고 있었는데 뜻도 좋아서 블로그 이름으로 쓸 것이다. 근데 조금 심심한 것 같아서 그 뒤를 더 추가해보았다.

<br>
<br>

`첫 게시글이라 간단하고 재미있게 작성해보았는데, 다음 게시글부터는 공부한 내용과 관련된 프로젝트를 작성할 것이다. 머신러닝을 중점으로 진행될 예정이다.`